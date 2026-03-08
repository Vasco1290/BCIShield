import torch
import torch.nn as nn
import time
from typing import Callable

def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculates classification accuracy.
    
    Args:
        outputs (torch.Tensor): Logits from the model of shape (batch_size, num_classes).
        labels (torch.Tensor): True labels of shape (batch_size,).
        
    Returns:
        float: Accuracy in percentage (0.0 to 100.0).
    """
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return (correct / total) * 100.0

def measure_latency(model: nn.Module, input_tensor: torch.Tensor, num_runs: int = 100, device: torch.device = torch.device('cpu')) -> float:
    """
    Measures the inference latency of a model.
    
    Args:
        model (nn.Module): The model to evaluate.
        input_tensor (torch.Tensor): Dummy input tensor of correct shape.
        num_runs (int): Number of inference runs to average over.
        device (torch.device): Device to run on.
        
    Returns:
        float: Average latency in milliseconds.
    """
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
            
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000.0) # Convert to ms
            
    return sum(latencies) / len(latencies)

def measure_defense_latency(model: nn.Module, defense_module: nn.Module, input_tensor: torch.Tensor, num_runs: int = 100, device: torch.device = torch.device('cpu')) -> float:
    """
    Measures inference latency including a preprocessing defense like Input Smoothing.
    
    Args:
        model (nn.Module): The model to evaluate.
        defense_module (nn.Module): The preprocessing defense module.
        input_tensor (torch.Tensor): Dummy input tensor.
        num_runs (int): Number of runs.
        device (torch.device): Computing device.
        
    Returns:
        float: Average latency in milliseconds.
    """
    model.eval()
    defense_module.eval()
    
    model.to(device)
    defense_module.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            x = defense_module(input_tensor)
            _ = model(x)
            
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            x = defense_module(input_tensor)
            _ = model(x)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000.0)
            
    return sum(latencies) / len(latencies)

def defense_effectiveness(clean_acc: float, adv_acc: float, def_adv_acc: float) -> float:
    """
    Calculates the effectiveness of a defense (accuracy recovery percentage).
    
    Effectiveness = (Defense Accuracy - Attack Accuracy) / (Clean Accuracy - Attack Accuracy) * 100
    If Clean == Attack, returns 0.0.
    
    Args:
        clean_acc (float): Accuracy on clean data.
        adv_acc (float): Accuracy under attack without defense.
        def_adv_acc (float): Accuracy under attack with defense.
        
    Returns:
        float: Effectiveness percentage.
    """
    if clean_acc == adv_acc:
        return 0.0
    recovery = (def_adv_acc - adv_acc) / (clean_acc - adv_acc)
    # Clip to max 1.0 (100%) and min 0.0
    recovery = max(0.0, min(1.0, recovery))
    return recovery * 100.0
