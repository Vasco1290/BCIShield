import pytest
import torch
from src.defenses.input_smoothing import GaussianSmoothing
from src.models.eegnet import EEGNet
from src.defenses.adversarial_training import train_adversarial_epoch
from src.attacks.fgsm import fgsm_attack
import torch.optim as optim

def test_gaussian_smoothing_shape():
    channels = 22
    samples = 1000
    batch_size = 2
    
    smoothing = GaussianSmoothing(channels=channels, kernel_size=5, sigma=1.0)
    x = torch.randn(batch_size, 1, channels, samples)
    
    out = smoothing(x)
    assert out.shape == x.shape, "Smoothing should preserve input shape"
    
def test_gaussian_smoothing_invalid_kernel():
    with pytest.raises(ValueError):
        # Even kernel size should raise error
        GaussianSmoothing(channels=22, kernel_size=4, sigma=1.0)

def test_adversarial_training_epoch():
    model = EEGNet()
    # Mock dataloader
    x = torch.randn(4, 1, 22, 1000)
    y = torch.tensor([0, 1, 2, 3])
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cpu')
    
    # Should run without crashing and return loss, acc
    loss, acc = train_adversarial_epoch(model, dataloader, optimizer, criterion, fgsm_attack, device, epsilon=0.01)
    
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
