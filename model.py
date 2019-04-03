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
        # Avg Pool the layers
        x = F.avg_pool2d(x,2)
        # Leaky ReLU and Convolution pair 1
        x = F.leaky_relu(self.conv1(x),negative_slope=0.1)
        # Leaky ReLU and Convolution pair 2
        x = F.leaky_relu(self.conv2(x),negative_slope=0.1)

        return x

class up(nn.Module):
    """
    This is going to be used for the creating of the neural network block.
    Based on the paper this is going to be an implementation of UNET so the layers would be:

    Bilinear Interpolation > Conv + Leak ReLU > Conv + Leaky ReLU

    This is the most basic block for the UNET that is going to be upsampling (deep to flat). Expecting this to be reused again on the network class.

    ----------------------------------------
    METHODS:

    forward(x)
        This would return the output tensor after passing input `x` to the block
    """
    def __init__(self, inChannels, outChannels):
        """
        Defines the structure of the convolutional layers per block. There are two conv layers for a basic block, one following the other.

        Parameters
        ----------------------------------------
        inChannels: int
            The number of input channels in the first convolutional layer

        outChannels: int
            The number of output channels for the first convolutional layer.
            For the 2nd conv layer this number is doubled to increase the size of the image

        """
        super(up, self).__init__()
        
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride = 1, padding = 1) # First block, Connected to input
        self.conv2 = nn.Conv2d(2*outChannels, outChannels, 3, stride = 1, padding = 1) # Second block, Connected to conv1

    def forward(self, x, skpCn):
        """
        This would return the output tensor after passing input `x` to the block

        Parameters
        ----------------------------------------
        x: tensor
            The input to the NN block. In this case the image/frame.
        skpCn : tensor
            This is the skip connection between the input and this NN block
        
        Returns
        ----------------------------------------
        tensor
            Output tensor after passing the input `x` through the NN block.

        """
        # Avg Pool the layers
        x = F.interpolate(x, scale_factor=2,mode='billinear')
        # Leaky ReLU and Convolution pair 1
        x = F.leaky_relu(self.conv1(x),negative_slope=0.1)
        # Leaky ReLU and Convolution pair 2 (introduction of skip connection)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn),1)),negative_slope=0.1)

        return x

class UNET(nn.Module):
    """
    This would be the implementation of the UNET architecture described in the original Super SloMo paper.
    ----------------------------------------
    METHODS:
    forward(x)
        Returns the output tensor after passing input `x` through the entire UNET architecture
    """
    def __init__(self, inChannels, outChannels):
        """
        I am defining the layers that would be used to create the UNET. The parameters are based on the paper including the filter sizes.

        Parameters
        ----------------------------------------
        inChannels: int
            The number of input channels in the UNET (based on input image?)

        outChannels: int
            The number of output channels for the UNET

        """
        super(UNET,self).__init__()
        # Definition of the entire UNET Architecture based on the original paper
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride = 1, padding = 3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride = 1, padding = 3)
        self.down1 = down(32, 64, 5) # First filter size is 5x5 to have a longer range in catching motion
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        # End of contraction so we start going up and expand
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.up5 = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, stride = 1, padding = 1)

    def forward(self, x):
        """
        Returns the output tensor after passing input `x` to the UNET.
        We are now going to link up all the layers we have initialized so they form up the UNET architecture.
        Parameters
        ----------------------------------------
        x: tensor
            The input to the entire UNET. So this would be the frames
        Returns
        ----------------------------------------
        tensor
            The output of the UNET. This should be the interpolated frame.
        """
        # Compression phase, downsampling
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        S1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        S2 = self.down1(S1)
        S3 = self.down2(S2)
        S4 = self.down3(S3)
        S5 = self.down4(S4)
        u = self.down5(S5)
        # Expansion phase, upsampling.
        # NOTE: We also have to add back the skip connections every up sample block.
        u = self.up1(u, S5)
        u = self.up2(u, S4)
        u = self.up3(u, S3)
        u = self.up4(u, S2)
        u = self.up5(u, S1)
        x - F.leaky_relu(self.conv3(u), negative_slope=0.1)
        return x


# At this stage we have completed defining our UNET architecture. The succeeding functions are the additional steps
# defined in the Super SlowMo paper that helped them achieve their results. This would include getting the warp coefficients.
# TODO: Complete the backwarp class, getFlowCoeff function and getWarpCoeff function
class backwarp(nn.Module):

    def __init__(self):
        return super().__init__()
    
    def forward(self, *input):
        return super().forward(*input)
    
# NOTE: This is the t values for the intermediate frames. There would be 7 frames between t=0 --> t=1
# We are using linspace to genrate the values instead of manually defining them. 1/8 = 0.125
t = np.linspace(0.125, 0.875, 7)

def getFlowCoeff (indices, device):
    pass
def getWarpCoeff(indices, device):
    pass