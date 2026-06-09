import pytest
import torch
import torch.nn as nn
from src.models.eegnet import EEGNet
from src.defenses.input_smoothing import GaussianSmoothing
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack

def test_adaptive_attack_gradient_propagation():
    """
    Tests that gradients propagate through the entire pipeline:
    GaussianSmoothing -> EEGNet -> Loss, and successfully produce bounded perturbations.
    """
    channels = 22
    samples = 1000
    batch_size = 2
    epsilon = 0.05
    
    # 1. Initialize modules
    model = EEGNet(num_classes=4, channels=channels, samples=samples)
    smoothing = GaussianSmoothing(channels=channels, kernel_size=5, sigma=1.0)
    
    # Composite model (Adaptive pipeline)
    adaptive_model = nn.Sequential(smoothing, model)
    
    # 2. Prepare dummy inputs
    x = torch.randn(batch_size, 1, channels, samples, requires_grad=True)
    y = torch.tensor([0, 2])
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate clean output
    clean_out = adaptive_model(x)
    loss = criterion(clean_out, y)
    loss.backward()
    
    # Verify gradient flows back to raw input x in baseline backward pass
    assert x.grad is not None, "Gradient did not flow back to input tensor x"
    assert torch.any(x.grad != 0.0), "Gradient is zero"
    
    # 3. Generate adaptive FGSM attack
    x_adv_fgsm, pert_fgsm = fgsm_attack(adaptive_model, x.detach(), y, epsilon, criterion)
    
    assert x_adv_fgsm.shape == x.shape
    assert pert_fgsm.shape == x.shape
    assert torch.max(torch.abs(x_adv_fgsm - x.detach())).item() <= epsilon + 1e-5
    
    # 4. Generate adaptive PGD attack
    x_adv_pgd, pert_pgd = pgd_attack(adaptive_model, x.detach(), y, epsilon, steps=5, criterion=criterion)
    
    assert x_adv_pgd.shape == x.shape
    assert pert_pgd.shape == x.shape
    assert torch.max(torch.abs(x_adv_pgd - x.detach())).item() <= epsilon + 1e-4
