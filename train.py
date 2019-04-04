# Creating the training block for the model

# Importing dependencies
import argparse
import torch
import torchvision
import torachvison.transforms as tranforms
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
import model
import dataloader
from math import log10
import datetime
from tensorboardX import SummaryWritter