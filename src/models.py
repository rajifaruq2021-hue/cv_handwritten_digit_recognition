"""
models.py

I define the neural network architectures for my Handwritten Digit Recognition project.
Course: DLBAIPCV01 - Project: Computer Vision
Author: Faruq Ishola Raji
Matriculation Number: 4253139

In this file, I implement and compare two different Convolutional Neural Networks (CNNs):
1. CholletModernConvNet:
   - I re-implement the modern baseline from Francois Chollet's 2021 book "Deep Learning with Python".
   - It takes 28x28 grayscale images.
   - It uses two convolution stages with 3x3 filters, ReLU activations, and 2x2 Max-Pooling.
   - It uses a 50% Dropout layer to prevent overfitting before the final 10-digit classification layer.

2. LeCun1989HistoricalCNN:
   - I reconstruct the historical 1989 network by Yann LeCun and his team.
   - It uses 5x5 filters with stride 2 for spatial sub-sampling (instead of max-pooling).
   - It uses Tanh (hyperbolic tangent) activations, which was the standard symmetric activation in 1989.
   - It includes a 25% Dropout layer between H2 and H3 to emulate the historical sparse connectivity.
   - It uses an intermediate hidden layer of 30 units before the final 10-digit output layer.
"""

# I import PyTorch and its neural network modules
import torch
import torch.nn as nn
import torch.nn.functional as F


class CholletModernConvNet(nn.Module):
    """
    I represent the modern ConvNet baseline from Chollet (2021).
    I use this model to see how modern design choices perform on MNIST digits.
    """

    def __init__(self, num_classes: int = 10):
        # I call the parent constructor from PyTorch nn.Module
        super(CholletModernConvNet, self).__init__()

        # First Convolution Block:
        # I take 1 input channel (grayscale) and produce 32 feature maps using 3x3 kernels
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0)

        # I use a 2x2 max-pooling window with stride 2 to cut the width and height in half
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolution Block:
        # I take the 32 feature maps and expand them to 64 feature maps using 3x3 kernels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)

        # I apply another 2x2 max-pooling layer to downsample the feature maps again
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Regularization:
        # I randomly drop 50% of the hidden connections during training so the model does not overfit
        self.dropout = nn.Dropout(p=0.5)

        # Final Classifier:
        # After two conv and pool stages, the spatial size is 5x5 with 64 channels (64 * 5 * 5 = 1600 numbers)
        # I map these 1,600 values directly into the 10 digit classes (digits 0 to 9)
        self.fc = nn.Linear(64 * 5 * 5, num_classes)

    def forward(self, x):
        """
        I define how the image passes forward through each layer of the modern network.
        """
        # I apply the first convolution and then pass the result through a ReLU activation
        x = F.relu(self.conv1(x))

        # I downsample using max-pooling (reduces size from 26x26 to 13x13)
        x = self.pool1(x)

        # I apply the second convolution and another ReLU activation
        x = F.relu(self.conv2(x))

        # I downsample again (reduces size from 11x11 to 5x5)
        x = self.pool2(x)

        # I flatten the 3D feature cube into a 1D vector of 1,600 numbers
        x = torch.flatten(x, 1)

        # I apply dropout during training
        x = self.dropout(x)

        # I compute the final 10 output scores
        x = self.fc(x)

        return x


class LeCun1989HistoricalCNN(nn.Module):
    """
    I represent the historical 1989 CNN architecture introduced by Yann LeCun et al.
    I reconstructed this model to evaluate how early deep learning designs compare to modern ones.
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.25):
        # I call the parent constructor
        super(LeCun1989HistoricalCNN, self).__init__()

        # Layer H1:
        # In 1989, LeCun used 5x5 filters and stride 2 to do convolution and downsampling in one single step.
        # I take 1 input channel and produce 12 feature maps.
        # An input of 28x28 with 5x5 filter and stride 2 produces a 12x12 output grid.
        self.h1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=2, padding=0)

        # Layer H2:
        # I apply another 12 filters of size 5x5 with stride 2.
        # This reduces the 12x12 feature maps down to 4x4.
        self.h2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=2, padding=0)

        # Sparse Connectivity Approximation:
        # In the original 1989 paper, H2 was not fully connected to H3; each unit only looked at a small group of maps.
        # To approximate this historical constraint on modern hardware, I use a 25% dropout layer here.
        self.dropout = nn.Dropout(p=dropout_rate)

        # Layer H3:
        # The 12 feature maps of size 4x4 give 12 * 4 * 4 = 192 features.
        # I connect these 192 features to 30 hidden units as specified in LeCun's paper.
        self.h3 = nn.Linear(12 * 4 * 4, 30)

        # Layer Output:
        # I connect the 30 hidden units to the 10 output classes for digits 0 through 9.
        self.output = nn.Linear(30, num_classes)

    def forward(self, x):
        """
        I define how the image passes forward through the historical network.
        """
        # I pass the image through H1 and apply the Tanh activation function
        x = torch.tanh(self.h1(x))

        # I pass through H2 and apply Tanh again
        x = torch.tanh(self.h2(x))

        # I flatten the feature maps into a 1D vector of 192 numbers
        x = torch.flatten(x, 1)

        # I apply dropout to simulate historical sparse connections
        x = self.dropout(x)

        # I pass through the 30-unit hidden layer H3 with Tanh activation
        x = torch.tanh(self.h3(x))

        # I produce the final 10 output scores
        x = self.output(x)

        return x


def count_parameters(model: nn.Module) -> int:
    """
    I count the total number of trainable weights and biases in any given PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# I run a small sanity check when this script is executed directly
if __name__ == '__main__':
    # I create one instance of each model
    chollet_net = CholletModernConvNet()
    lecun_net = LeCun1989HistoricalCNN()

    # I print their parameter counts so I can compare their sizes
    print("Chollet (2021) Modern ConvNet trainable parameters:", count_parameters(chollet_net))
    print("LeCun et al. (1989) Historical CNN trainable parameters:", count_parameters(lecun_net))

    # I pass dummy images through both networks to make sure the shapes match
    dummy_input = torch.randn(2, 1, 28, 28)
    print("Chollet output tensor shape:", chollet_net(dummy_input).shape)
    print("LeCun output tensor shape:", lecun_net(dummy_input).shape)
    print("Sanity check passed successfully!")
