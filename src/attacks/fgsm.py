import torch
import torch.nn as nn

def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epsilon: float, criterion: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implements the Fast Gradient Sign Method (FGSM) attack.
    
    A single-step attack that adds a small perturbation to the input image in the direction 
    of the gradient of the loss function with respect to the input.
    Equation: x_adv = x + epsilon * sign(gradient)
    
    Args:
        model (nn.Module): The classification model being attacked (EEGNet).
        x (torch.Tensor): Clean input tensor of shape (batch, 1, channels, samples).
        y (torch.Tensor): True labels for the input.
        epsilon (float): The magnitude of the perturbation (L-infinity norm bound).
        criterion (nn.Module): The loss function (e.g., CrossEntropyLoss).
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - Perturbed input tensor (x_adv)
            - Perturbation added (perturbation)
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Enable gradient calculation for the input
    x_adv = x.clone().detach().to(x.device).requires_grad_(True)
    y = y.to(x.device)
    
    # Forward pass
    outputs = model(x_adv)
    loss = criterion(outputs, y)
    
    # Calculate gradients of the model in backward pass
    model.zero_grad()
    loss.backward()
    
    # Collect gradient
    data_grad = x_adv.grad.data

    # FGSM: x_adv = x + epsilon * sign(gradient)
    perturbation = epsilon * data_grad.sign()

    # Apply perturbation to original input (not x_adv)
    perturbed_x = x.detach() + perturbation

    # Enforce L-infinity epsilon-ball constraint
    perturbed_x = torch.clamp(perturbed_x, x - epsilon, x + epsilon)

    # Recalculate true perturbation after clipping
    perturbation = (perturbed_x - x).detach()

    return perturbed_x.detach(), perturbation
