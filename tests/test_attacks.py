import pytest
import torch
import torch.nn as nn
from src.models.eegnet import EEGNet
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack

@pytest.fixture
def setup_model_data():
    model = EEGNet()
    x = torch.randn(2, 1, 22, 1000)
    y = torch.tensor([0, 1])
    criterion = nn.CrossEntropyLoss()
    return model, x, y, criterion

def test_fgsm_attack(setup_model_data):
    model, x, y, criterion = setup_model_data
    epsilon = 0.1
    
    x_adv, perturbation = fgsm_attack(model, x, y, epsilon, criterion)
    
    assert x_adv.shape == x.shape
    assert perturbation.shape == x.shape
    
    # Check that perturbation magnitude is bounded by epsilon
    # Max absolute difference between x and x_adv
    max_diff = torch.max(torch.abs(x_adv - x)).item()
    assert max_diff <= epsilon + 1e-5 # add small tolerance for float precision

def test_pgd_attack(setup_model_data):
    model, x, y, criterion = setup_model_data
    epsilon = 0.1
    steps = 5
    
    x_adv, perturbation = pgd_attack(model, x, y, epsilon, steps, criterion)
    
    assert x_adv.shape == x.shape
    assert perturbation.shape == x.shape
    
    max_diff = torch.max(torch.abs(x_adv - x)).item()
    assert max_diff <= epsilon + 1e-4
