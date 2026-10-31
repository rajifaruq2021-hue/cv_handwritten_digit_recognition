"""
Neural Network Architectures for Handwritten Digit Recognition
Course: DLBAIPCV01 - Project: Computer Vision
Author: Faruq Ishola Raji (4253139)

Implements two Convolutional Neural Network architectures:
1. Chollet (2021) Modern Keras ConvNet (re-implemented in PyTorch for clean execution):
   - Input: (1, 28, 28)
   - Conv2d(32 filters, 3x3 kernel, ReLU)
   - MaxPool2d(2x2)
   - Conv2d(64 filters, 3x3 kernel, ReLU)
   - MaxPool2d(2x2)
   - Flatten
   - Dropout(0.5)
   - Linear(1600 -> 10, Softmax)

2. LeCun et al. (1989) Historical 4-Layer Backpropagation Network Approximation:
   - Input: (1, 28, 28)
   - Layer H1: Conv2d(1 -> 12 feature maps, 5x5 kernel, stride 2, no padding) -> output (12, 12, 12)
   - Activation: Tanh (LeCun 1989 used symmetric sigmoid / tanh)
   - Layer H2: Conv2d(12 -> 12 feature maps, 5x5 kernel, stride 2, no padding) -> output (12, 4, 4)
   - Activation: Tanh
   - Dropout(0.25): Emulates the historical constrained sparse connections between H2 and H3 as suggested by IU
   - Layer H3: Linear(12 * 4 * 4 = 192 -> 30 hidden units, fully connected)
   - Activation: Tanh
   - Layer Output: Linear(30 -> 10 units)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CholletModernConvNet(nn.Module):
    """
    Chollet (2021) Modern ConvNet architecture for MNIST.
    """
    def __init__(self, num_classes: int = 10):
        super(CholletModernConvNet, self).__init__()
        # Conv layer 1: 1 -> 32, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv layer 2: 32 -> 64, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Classification head
        self.dropout = nn.Dropout(0.5)
        # 28x28 -> conv1 (26x26) -> pool1 (13x13) -> conv2 (11x11) -> pool2 (5x5) -> 64 * 5 * 5 = 1600
        self.fc = nn.Linear(64 * 5 * 5, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class LeCun1989HistoricalCNN(nn.Module):
    """
    Approximation of the historical LeCun et al. (1989) backpropagation network.
    Uses Tanh activations, 5x5 stride-2 receptive fields, and Dropout to approximate
    the historical sparse layer connections.
    """
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.25):
        super(LeCun1989HistoricalCNN, self).__init__()
        # Layer H1: 12 feature maps, 5x5 kernel, stride 2 (undersampling)
        # 28x28 -> (28 - 5)/2 + 1 = 12x12
        self.h1 = nn.Conv2d(1, 12, kernel_size=5, stride=2, padding=0)
        
        # Layer H2: 12 feature maps, 5x5 kernel, stride 2
        # 12x12 -> (12 - 5)/2 + 1 = 4x4
        self.h2 = nn.Conv2d(12, 12, kernel_size=5, stride=2, padding=0)
        
        # Dropout to emulate the historical constrained sparse connectivity between H2 and H3
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer H3: 30 fully connected hidden units
        # Flattened H2 size = 12 maps * 4 * 4 = 192 inputs
        self.h3 = nn.Linear(12 * 4 * 4, 30)
        
        # Layer Output: 10 output units for digits 0-9
        self.output = nn.Linear(30, num_classes)

    def forward(self, x):
        # Tanh activation as used in LeCun et al. (1989)
        x = torch.tanh(self.h1(x))
        x = torch.tanh(self.h2(x))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = torch.tanh(self.h3(x))
        x = self.output(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    chollet_net = CholletModernConvNet()
    lecun_net = LeCun1989HistoricalCNN()
    
    print("Chollet (2021) Modern ConvNet trainable parameters:", count_parameters(chollet_net))
    print("LeCun et al. (1989) Historical CNN trainable parameters:", count_parameters(lecun_net))
    
    dummy_x = torch.randn(2, 1, 28, 28)
    print("Chollet forward output shape:", chollet_net(dummy_x).shape)
    print("LeCun forward output shape:", lecun_net(dummy_x).shape)
