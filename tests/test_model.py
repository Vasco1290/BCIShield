import pytest
import torch
from src.models.eegnet import EEGNet

def test_eegnet_forward_shape():
    batch_size = 4
    channels = 22
    samples = 1000
    num_classes = 4

    model = EEGNet(num_classes=num_classes, channels=channels, samples=samples)
    x = torch.randn(batch_size, 1, channels, samples)
    out = model(x)
    
    assert out.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {out.shape}"

def test_eegnet_backward_pass():
    model = EEGNet()
    x = torch.randn(2, 1, 22, 1000)
    y = torch.tensor([0, 1])
    criterion = torch.nn.CrossEntropyLoss()
    
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    
    # Check if gradients are populated
    for param in model.parameters():
        assert param.grad is not None
