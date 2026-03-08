import torch
import torch.nn as nn
from typing import Callable, Optional

def train_adversarial_epoch(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    attack_fn: Callable, 
    device: torch.device,
    epsilon: float = 0.1,
    alpha: float = 0.5
) -> tuple[float, float]:
    """
    Performs one epoch of adversarial training.
    
    Adversarial training mixes clean and adversarial examples during the training
    process to increase the model's robustness against adversarial perturbations.
    
    Args:
        model (nn.Module): The classification model being trained (EEGNet).
        dataloader (torch.utils.data.DataLoader): DataLoader providing the training data.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        attack_fn (Callable): Attack function to generate adversarial examples (e.g., fgsm_attack or pgd_attack).
        device (torch.device): Device to perform computations on (CPU or CUDA).
        epsilon (float): The perturbation magnitude to use for adversarial examples. Default is 0.1.
        alpha (float): Weight for combining clean loss and adversarial loss (0.0 to 1.0). 
                       Loss = alpha * clean_loss + (1 - alpha) * adv_loss. Default is 0.5.
                       
    Returns:
        tuple[float, float]: 
            - Average total loss for the epoch
            - Accuracy on the mixed (clean/adv) training data
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # We need to temporarily set the model to eval mode to generate adversarial examples properly
        # if the attack function expects it, but our attack functions handle that internally or we
        # ensure it's evaluated for the attack, then set back to train.
        # However, to avoid messing up BatchNorm stats during attack generation, many implementations
        # set model.eval() during attack generation, then model.train() for the actual update.
        model.eval()
        
        # Generate adversarial examples
        # attack_fn is expected to return (x_adv, perturbation)
        inputs_adv, _ = attack_fn(model, inputs, labels, epsilon=epsilon, criterion=criterion)
        inputs_adv = inputs_adv.to(device)
        
        # Set back to train mode for the parameter update
        model.train()
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass for clean examples
        outputs_clean = model(inputs)
        loss_clean = criterion(outputs_clean, labels)
        
        # Forward pass for adversarial examples
        outputs_adv = model(inputs_adv)
        loss_adv = criterion(outputs_adv, labels)
        
        # Combine losses
        # Loss = alpha * clean_loss + (1 - alpha) * adv_loss
        loss = alpha * loss_clean + (1.0 - alpha) * loss_adv
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        
        # Calculate accuracy on the adversarial examples as a primary indicator of robustness
        _, predicted = torch.max(outputs_adv, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = total_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc
