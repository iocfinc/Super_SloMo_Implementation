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

# Parsers to get commandline arguments
# Will not mess with this one
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=True, help='path to dataset folder containing train-test-validation folders')
parser.add_argument("--checkpoint_dir", type=str, required=True, help='path to checkpoint directory during saving')
parser.add_argument("--checkpoint", type=str, help="path to checkpoint when using a pretrained model" )
parser.add_argument("--train_continue", type=bool, default=False, help="Used when resuming from a checkpoint, set to True and set the `--checkpoint` path. Default: False")
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs for training. Default: 200")
parser.add_argument("--train_batch_size", type=int, default=6, help="Batch size for training. Default:6")
parser.add_argument("--validation_batch_size", type=int, default=10, help="Batch size for validation. Default:10")
parser.add_argument("--init_learning_rate", type=float, default=0.0001, help="Set the initial learning rate. Default: 0.0001")
parser.add_argument("--milestones", type=list, default=[100,150], help="Set the epoch values where we decrease the learning rate by a factor of 0.1. Default: [100, 150]")
parser.add_argument("--progress_iter", type=int, default=100, help="Frequency of printing progress on training and validation. Default: every 100 iterations")
parser.add_argument("--checkpoint_epoch", type=int, default=5, help="Frequence of saving a checkpoint, every N. Default: every 5 epochs. 1 epoch ~ 151MB")
args = parser.parse_args()

# TODO: Check if I can use fast.ai's learning rate solver to get a better learning rate.

# For tensorboardX
## Visualizing loss and interpolated frames
writer = SummaryWriter('log')

## Initialization of flow computations

device = torch.device("cude:0" if torch.cuda.is_available() else "cpu") # Defaults always to CPU if GPU is unavailable
flowComp = model.UNET(6,4)
flowComp.to(device)
ArbTimeFlowInter = model.UNET(20,5)
ArbTimeFlowInter.to(device)

## Initialization of backward warpers on training and validation datasets

trainFlowBackWarp = model.backwarp(352,352,device)
trainFlowBackWarp = trainFlowBackWarp.to(device)
validationFlowBackWarp = model.backwarp(640,352,device)
validationFlowBackWarp = validationFlowBackWarp.to(device)

# Channel wise mean calculated on adobe240-fps training dataset
# This is for loading and transformation
mean = [0.429, 0.431, 0.397]
std = [1,1,1]
normalize = tranforms.Normalize(mean=mean, std=std)
transform = transforms.Compose([transofrms.ToTensor(), normalize])

trainset = dataloader.SuperSloMo(root=args.dataset_root + '/train', transforms=transform,train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True)

validationset = dataloader.SuperSloMo(root=*args.dataset_root +'/validation', transforms=transform, randomCropSize=(640, 352), train=False)
validationloader = torch.utils.data.DataLoader(validationset,batch_size = args.validation_batch_size, shuffle=False)

print(trainset,validationset)

### Transformations: Displaying image from tensor

negmean = [x * -1 for x in mean]
revNormalize = transforms.Normalize(mean=negmean, std=std)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

# Utils

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

## Optimizer and Loss initialization

L1_lossFn = nn.L1Loss()
MSE_LossFn = nn.MSELoss()

params = list(ArbTimeFlowInter.parameters())+list(flowComp.parameters())
optimizer = optim.Adam(prams, lr=args.init_learning_rate)

# Scheduler for Learning reate reduction
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = args.milestones, gamma=0.1)

## VGG16 model initialization
# This is based on the paper. We require only layer conv_4_3.

vgg16 = torchvision.models.vgg16()
vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
vgg16_conv_4_3.to(device)

for param in vgg16_conv_4_3.parameters():
    param.requires_grad = False

# Creating the validation function

def validate():
    # Initializing variables
    psnr = 0
    tloss = 0
    flag = 1
    with torch.no_grad():
        for validationIndex, (validationData, validationFrameIndex) in enumerate(validationloader,0):
            frame0, frameT, frame1 = validationData

            I0 = frame0.to(device)
            I1 = frame1.to(device)
            IFrame = frameT.to(device)

            flowOut = flowComp(torch.cat((I0,I1), dim = 1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            fCoeff = model.getFlowCoeff(validationFrameIndex, device)
            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            g_I0_F_t_0 = validationFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = validationFlowBackWarp(I1, F_t_1)

            intrpOut = ArbTimeFlowInter(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0 = F.sigmoid(intrpOut[:, 4:5, :, :])
            V_t_1 = 1 - V_t_0

            g_I0_F_t_0_f = validationFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = validationFlowBackWarp(I1, F_t_1_f)
            wCoeff = model.getWarpCoeff(validationFrameIndex, device)

            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f)/(wCoeff[0] * V_t_0 + wCoeff[0] * V_t_1)

            # Tensorboard flags
            if (flag):
                retImg = torchvision.utils.make_grid([revNormalize(frame0[0]), revNormalize(frameT[0]), revNormalize(Ft_p.cpu()[0]), revNormalize(frame1[0])], padding=10)

            # Losses
            recnLoss = L1_lossFn(Ft_p, IFrame)
            prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))
            warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) +L1_lossFn(validationFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(validationFlowBackWarp(I1, F_0_1), I0)

            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1: ,:]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1

            loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
            tloss += loss.item()

            #PSNR
            MSE_val = MSE_LossFn(Ft_p, IFrame)
            psnr += (10 * log10(1/MSE_val.item()))
    return (psnr / len(validationloader)), (tloss/ len(validationloader)), retImg

# Initialization of state dict. Will load or continue from a check point if train_continue flag is set.
# Basically, catching if we train from scratch or from a checkpoint.

if args.train_continue:
    dict1 = torch.load(args.checkpoint)
    ArbTimeFlowInter.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state(dict1['state_dictFC'])
else:
    dict1 = {'loss':[], 'valLoss':[], 'valPSNR':[], 'epoch':-1}




