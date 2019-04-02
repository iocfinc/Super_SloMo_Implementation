# Implementation of the model network from the paper.
# This is just retyping all the modules to learn more about what it does.
# Based on the paper this is an implementation of a UNET CNN.


# TODO: Import all the dependencies
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# TODO: define the downsampling portion as a class for reusability

class down(nn.Module):
    """
    This is going to be used for the creating of the neural network block.
    Based on the paper this is going to be an implementation of UNET so the layers would be:

    Average Pooling > Convolution + Leaky ReLU > Convolution + Leaky ReLU

    This is the most basic block for the UNET that is going to be downsampling (flat to wide). Exepecting this to be reused again on the newtok calss.

    ----------------------------------------
    METHODS:

    forward(x)
        This would return the output tensor after passing input `x` to the block
    """
    def __init__(self, inChannels, outChannels, filterSize):
        """
        Defines the structure of the convolutional layers per block. There are two conv layers for a basic block, one following the other.

        Parameters
        ----------------------------------------
        inChannels: int
            The number of input channels in the first convolutional layer

        outChannels: int
            The number of output channels for the first convolutional layer.
            For the second convolutional layer this is going to be the input dimmension as well since it is piped.

        filterSize: int
        The filter size for the convolutional block. This would create a (filterSize x filterSize) sized filter for both conv1 and conv2.

        """
        super(down, self).__init__()
        
        self.conv1 = nn.Conv2d(inChannels, outChannels, filterSize, stride = 1, padding = int(filterSize - 1)/2) # First block, Connected to input
        self.conv2 = nn.Conv2d(outChannels, outChannels, filterSize, stride = 1, padding = int(filterSize - 1)/2) # Second block, Connected to conv1

    def forward(self, x):
        """
        This would return the output tensor after passing input `x` to the block

        Parameters
        ----------------------------------------
        x: tensor
            The input to the NN block. In this case the image/frame.
        
        Returns
        ----------------------------------------
        tensor
            Output tensor after passing the input `x` through the NN block.

        """
        return x