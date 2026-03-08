import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianSmoothing(nn.Module):
    """
    Input Smoothing Defense using 1D Gaussian smoothing applied to time-series (EEG) data.
    
    Applying an input smoothing filter is a preprocessing step that acts as a defense
    mechanism. It helps to mitigate the high-frequency adversarial perturbations added 
    by attacks like FGSM or PGD by blurring the signal slightly before classifying.
    """
    
    def __init__(self, channels: int, kernel_size: int = 5, sigma: float = 1.0):
        """
        Initializes the Gaussian Smoothing module.
        
        Args:
            channels (int): Number of input channels (e.g., 22 for BCIC-IV-2a).
            kernel_size (int): Size of the Gaussian kernel. Should be odd. Default is 5.
            sigma (float): Standard deviation of the Gaussian kernel. Default is 1.0.
        """
        super(GaussianSmoothing, self).__init__()
        
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number.")
            
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # Create a 1D Gaussian kernel
        # Math: G(x) = (1 / (sqrt(2*pi) * sigma)) * exp(- (x^2) / (2 * sigma^2))
        kernel = self._create_gaussian_kernel1d(kernel_size, sigma)
        
        # Shape the kernel to be used in Conv2d as a depthwise convolution along the time axis
        # Input shape: (batch, 1, channels, samples) -> we apply 1D smoothing along 'samples' per channel.
        # But wait, Conv2d with group=channels over (batch, channels, 1, samples) is easier.
        # Since our input is (batch, 1, channels, samples), it behaves like a 2D image where H=channels, W=samples.
        # We can use a 2D kernel of shape (1, kernel_size) to smooth only over the time dimension.
        # So kernel shape for Conv2d: (out_channels=1, in_channels=1, 1, kernel_size)
        kernel_2d = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('weight', kernel_2d)
        
    def _create_gaussian_kernel1d(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Helper function to create a 1D Gaussian kernel."""
        # Create a coordinate grid ranging from -radius to +radius
        radius = kernel_size // 2
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        
        # Calculate unnormalized Gaussian values
        kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        
        # Normalize the kernel so that it sums to 1
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Gaussian smoothing to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 1, channels, samples).
            
        Returns:
            torch.Tensor: Smoothed input tensor of the same shape.
        """
        # Apply 2D convolution with a 1xkernel_size filter over the time dimension.
        # Padding applied only to time dimension (left and right) to keep the length same.
        padding = (self.kernel_size // 2, self.kernel_size // 2, 0, 0)
        x_padded = F.pad(x, padding, mode='reflect')
        
        # Convolve using the registered Gaussian weight
        smoothed_x = F.conv2d(x_padded, self.weight, stride=1, padding=0)
        
        return smoothed_x
