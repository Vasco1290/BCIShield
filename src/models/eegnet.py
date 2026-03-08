import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """
    EEGNet Implementation in PyTorch based on Lawhern et al., 2018 (Journal of Neural Engineering).
    
    A compact Convolutional Neural Network designed specifically for EEG data classification.
    It uses regular convolutions, depthwise convolutions, and separable convolutions to extract
    spectral and spatial features sequentially.
    """
    
    def __init__(self, num_classes: int = 4, channels: int = 22, samples: int = 1000, 
                 dropout_rate: float = 0.5, F1: int = 8, D: int = 2, F2: int = 16):
        """
        Initializes the EEGNet architecture.
        
        Args:
            num_classes (int): Number of target classes to predict. Default is 4 (e.g., motor imagery classes).
            channels (int): Number of input EEG channels. Default is 22.
            samples (int): Number of time samples per EEG trial. Default is 1000.
            dropout_rate (float): Probability for dropout layers. Default is 0.5.
            F1 (int): Number of temporal filters. Default is 8.
            D (int): Number of spatial filters per temporal filter. Default is 2.
            F2 (int): Number of pointwise filters. Default is 16 (usually F1 * D).
        """
        super(EEGNet, self).__init__()
        
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.num_classes = num_classes
        self.channels = channels
        self.samples = samples
        
        # Block 1: Temporal Convolution
        self.block1 = nn.Sequential(
            # temporal padding = (kernel_size - 1) // 2
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 31), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        
        # Block 2: Depthwise Spatial Convolution
        self.block2 = nn.Sequential(
            # depthwise convolution over channels
            nn.Conv2d(self.F1, self.F1 * self.D, (self.channels, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # Block 3: Separable Convolution
        self.block3 = nn.Sequential(
            # Separable conv is depthwise followed by pointwise
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16), padding=(0, 7), groups=self.F1 * self.D, bias=False),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # Dynamically calculate classifier input size by running a dummy 
        # forward pass through feature blocks — avoids hardcoded assumptions
        # about sample rate or pooling arithmetic
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, samples)
            dummy_out = self.block1(dummy)
            dummy_out = self.block2(dummy_out)
            dummy_out = self.block3(dummy_out)
            self._feature_size = dummy_out.flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._feature_size, num_classes)
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through feature extraction blocks only (no classifier)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of EEGNet.
        Args:
            x (torch.Tensor): Shape (batch, 1, channels, time_samples)
        Returns:
            torch.Tensor: Logits of shape (batch, num_classes)
        """
        x = self._forward_features(x)
        x = self.classifier(x)
        return x
