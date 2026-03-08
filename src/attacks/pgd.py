import torch
import torch.nn as nn

def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epsilon: float, steps: int = 10, criterion: nn.Module = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Implements the Projected Gradient Descent (PGD) attack.
    
    An iterative attack that applies FGSM multiple times with a smaller step size, 
    and after each step, projects the perturbed data back into the epsilon-ball 
    around the original input to ensure the total perturbation is bounded.
    
    Args:
        model (nn.Module): The classification model being attacked (EEGNet).
        x (torch.Tensor): Clean input tensor of shape (batch, 1, channels, samples).
        y (torch.Tensor): True labels for the input.
        epsilon (float): The total bounded magnitude of the perturbation (L-infinity norm).
        steps (int): The number of iterative steps. Default is 10.
        criterion (nn.Module): The loss function. Default is CrossEntropyLoss.
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - Perturbed input tensor (x_adv)
            - Total perturbation added
    """
    # Avoid mutable default argument — instantiate criterion if not provided
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    
    # Iterative step-size calculation (alpha = epsilon / steps * 2.5)
    # The factor 2.5 is commonly used to ensure we can reach boundaries of the epsilon ball.
    # E.g. in Madry et al. (2018).
    alpha = (epsilon / steps) * 2.5
    
    # Initialize with random perturbation within epsilon ball on correct device
    perturbation = torch.empty_like(x).uniform_(-epsilon, epsilon).to(x.device)
    y = y.to(x.device)
    x_adv = (x + perturbation).detach()
    
    for _ in range(steps):
        x_adv.requires_grad = True
        
        # Forward pass
        outputs = model(x_adv)
        loss = criterion(outputs, y)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update the adversarial example
        # x_adv = x_adv + alpha * sign(gradient)
        data_grad = x_adv.grad.data
        x_adv = x_adv.detach() + alpha * data_grad.sign()
        
        # Projection step: ensuring the perturbation is within epsilon ball
        total_perturbation = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = x + total_perturbation
        x_adv = x_adv.detach()
        
    final_perturbation = x_adv - x
    return x_adv, final_perturbation
