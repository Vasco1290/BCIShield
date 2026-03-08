import os
import yaml
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

from src.models.eegnet import EEGNet
from src.data.dataset import get_dataloaders
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack
from src.defenses.input_smoothing import GaussianSmoothing
from src.defenses.adversarial_training import train_adversarial_epoch
from src.evaluation.metrics import calculate_accuracy, measure_latency, measure_defense_latency, defense_effectiveness

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model, dataloader, device, attack_fn=None, epsilon=0.0, 
                       defense_module=None, attack_kwargs=None):
    model.eval()
    if defense_module:
        defense_module.eval()
        
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # We need attacks bounded inside the defense logic if evaluating defense vs attack
        # Standard: attacker knows defense (white-box) or doesn't. 
        # Here we do naive evaluation: Attack the model, then apply defense to the attacked sample.
        
        if attack_fn is not None and epsilon > 0.0:
            kwargs = attack_kwargs or {}
            inputs_adv, _ = attack_fn(
                model, inputs, labels, 
                epsilon=epsilon, 
                criterion=criterion,
                **kwargs
            )
        else:
            inputs_adv = inputs
            
        model.eval()
        with torch.no_grad():
            x = inputs_adv
            if defense_module is not None:
                x = defense_module(x)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return (correct / total) * 100.0 if total > 0 else 0.0

def train_base_model(model, train_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for __ in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def run_experiment_for_subject(subject_id, config, device, results_list):
    print(f"\n--- Running Experiment for Subject {subject_id} ---")
    data_dir = config['dataset']['path']
    batch_size = config['training']['batch_size']
    
    # Get dataloaders
    try:
        train_loader, test_loader = get_dataloaders(data_dir, subject_id, batch_size)
    except Exception as e:
        print(f"Skipping Subject {subject_id} due to dataloader error (e.g., missing data): {e}")
        return

    # Base Model Initialization and Training
    model = EEGNet(num_classes=config['dataset']['classes'], channels=config['dataset']['channels'], samples=config['dataset']['samples']).to(device)
    print("Training Base Model...")
    model = train_base_model(model, train_loader, config['training']['epochs'], config['training']['learning_rate'], device)
    
    # 1. Base Evaluation (Clean)
    print("Evaluating Clean Accuracy...")
    clean_acc = evaluate_model(model, test_loader, device)
    
    # Measure Latency (Clean)
    dummy_input = torch.randn(1, 1, config['dataset']['channels'], config['dataset']['samples']).to(device)
    base_latency = measure_latency(model, dummy_input, device=device)
    
    results_list.append({
        'subject_id': subject_id, 'attack': 'None', 'epsilon': 0.0, 
        'defense': 'None', 'accuracy': clean_acc, 'latency_ms': base_latency
    })

    # Prepare Defense Modules
    smoothing = GaussianSmoothing(
        channels=config['dataset']['channels'], 
        kernel_size=config['defenses']['input_smoothing']['kernel_size'],
        sigma=config['defenses']['input_smoothing']['sigma']
    ).to(device)
    smoothing_latency = measure_defense_latency(model, smoothing, dummy_input, device=device)
    
    # Train Adv Model
    print("Training Adversarially Robust Model...")
    adv_model = copy.deepcopy(model).to(device)
    optimizer_adv = optim.Adam(adv_model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    adv_epochs = config['training']['adv_epochs']
    train_eps = config['defenses']['adversarial_training']['train_epsilon']
    train_alpha = config['defenses']['adversarial_training']['alpha']
    
    for __ in range(adv_epochs):
        train_adversarial_epoch(adv_model, train_loader, optimizer_adv, criterion, pgd_attack, device, epsilon=train_eps, alpha=train_alpha)
        
    adv_model_latency = base_latency # same architecture, same latency
        
    attacks = [
        ('FGSM', fgsm_attack, config['attacks']['fgsm']['epsilons'], {}),
        ('PGD', pgd_attack, config['attacks']['pgd']['epsilons'], 
         {'steps': config['attacks']['pgd']['steps']})
    ]
               
    for attack_name, attack_fn, epsilons, attack_kwargs in attacks:
        for eps in epsilons:
            print(f"Evaluating {attack_name} Epsilon {eps}...")
            # 2. Attack vs No Defense
            adv_acc = evaluate_model(model, test_loader, device, 
                                     attack_fn=attack_fn, epsilon=eps,
                                     attack_kwargs=attack_kwargs)
            results_list.append({
                'subject_id': subject_id, 'attack': attack_name, 'epsilon': eps, 
                'defense': 'None', 'accuracy': adv_acc, 'latency_ms': base_latency
            })
            
            # 3. Attack vs Input Smoothing
            smooth_acc = evaluate_model(model, test_loader, device, 
                                        attack_fn=attack_fn, epsilon=eps, 
                                        defense_module=smoothing,
                                        attack_kwargs=attack_kwargs)
            results_list.append({
                'subject_id': subject_id, 'attack': attack_name, 'epsilon': eps, 
                'defense': 'Input Smoothing', 'accuracy': smooth_acc, 'latency_ms': smoothing_latency
            })
            
            # 4. Attack vs Adversarial Training
            adv_train_acc = evaluate_model(adv_model, test_loader, device, 
                                           attack_fn=attack_fn, epsilon=eps,
                                           attack_kwargs=attack_kwargs)
            results_list.append({
                'subject_id': subject_id, 'attack': attack_name, 'epsilon': eps, 
                'defense': 'Adversarial Training', 'accuracy': adv_train_acc, 'latency_ms': adv_model_latency
            })

    # Save incremental results after each subject — prevents data loss on crash
    if len(results_list) > 0:
        os.makedirs(os.path.join('results', 'tables'), exist_ok=True)
        incremental_path = os.path.join('results', 'tables', 'results_incremental.csv')
        pd.DataFrame(results_list).to_csv(incremental_path, index=False)
        print(f"  Incremental results saved to {incremental_path}")

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'default_config.yaml')
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = []
    
    for sub in config['dataset']['subjects']:
        run_experiment_for_subject(sub, config, device, results)
        
    if len(results) > 0:
        df = pd.DataFrame(results)
        os.makedirs(os.path.join('results', 'tables'), exist_ok=True)
        out_csv = os.path.join('results', 'tables', 'experiment_results.csv')
        df.to_csv(out_csv, index=False)
        print(f"\nExperiments complete. Results saved to {out_csv}")
        print(df.head())
    else:
        print("\nNo results generated (data might be missing).")

if __name__ == "__main__":
    main()
