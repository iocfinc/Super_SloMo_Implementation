import argparse
import os
import os.path
import ctypes
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import model
import dataloader
import platform
from tqdm import tqdm

# Start up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_dir",type=str,help="Path to ffmpeg.exe")
parser.add_argument("--video",type=str, required=True, help="Path of the input video (video to be converted)")
parser.add_argument("--checkpoint",type=str,required=True, help="Path to the checkpoint of pretrained model")
parser.add_argument("--fps",type=float, default=30,help="FPS of the output video. Default: 30")
parser.add_argument("--sf",type=int, required=True,help="Slomo factor N. Increase the frames N times. ")
parser.add_argument("--batch_size",type=int,default=1, help="Specify the batch size for conversion. Will eat up memory but could decrease processing time. Default:1")
parser.add_argument("--output",type=str, default="output.mp4",help="Specify the output filename. Default: output.mp4")
args = parser.parse_args()

# Error handling

def check():
    """
    Basic Error Handling
    Returns
    ----------------------------------------
    error: string
        Indicates the error that was detected if any. Otherwise blank.
    """
    error = ''
    if (args.sf <2):
        error = "Error: --sf/slomo factor has to be at least 2"
    if (args.batch_size <1):
        error = "Error: --batch_size has to be at least 1"
    if (args.fps < 1):
        error = "Error: --fps has to be at least 1"
    return error
def extract_frames(video, outDir):
    """
    Process the input video to its fram components for processing.
    
    Parameters
    ----------------------------------------
    video: string
        The path of the video file to be converted.
    outDir: string
        The path of the directory where the extracted frames would be placed.
    
    Returns
    ----------------------------------------
    error: string
        Returns an error message if there are other errors.
    """
    error = ''
    print('ffmpeg -i {} -vsync 0 -qscale:v 2 {}/%06d.jpg'.format(video, outDir))
    retn = os.system('{} -i {} -vsync 0 -qscale:v 2 {}/%06d.jpg'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), video, outDir))
    if retn:
        error = "Error converting file {}. Exiting".format(video)
    return error
def create_video(dir):
    error = ''
    print('ffnoeg -r {} -i {}/%d.jpg =qscale:v 2 {}'.format(args.fps, dir, args.output))
    retn = os.system('{} -r {} -i {}/%d.jpg =crf -vcodec libx264 {}'.format(os.path.join(args.ffmpeg_dir, 'ffmpeg'), args.fps, dir, args.output))
    if retn:
        error = "Error creating output video. Exiting"
    return error

def main():
    error = check()
    if error:
        print(error)
        exit(1)
    IS_WINDOWS = 'Windows' == platform.system()
    extractionDir = 'tmpSuperSloMo'
    if not IS_WINDOWS:
        extractionDir = "." + extractionDir
    if os.path.isdir(extractionDir):
        rmtree(extractionDir)
    os.makedir(extractionDir)
    if IS_WINDOWS:
        FILE_ATTRIBUTE_HIDDEN = 0x02
        ctypes.windll.kernel32.SetFileAttributesW(extractionDir, FILE_ATTRIBUTE_HIDDEN)

    extractionPath = os.path.join(extractionDir, "input")
    outputPath  = os.path.join(extractionDir, 'output')
    os.makedirs(extractionPath, exist_ok=True)
    os.makedirs(outputPath, exist_ok=True)
    error = extract_frames(args.video, extractionPath)
    if error:
        print(error)
        exit(1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mean = [0.429, 0.431, 0.397]
    std = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean, std=std)
    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)
    
    if (device == 'cpu'):
        # This is from the source repo.
        transform = transforms.Compose([transforms.ToTensor()])
        TP = transforms.Compose([transforms.ToPILImage()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        TP = transforms.Compose([revNormalize, transforms.ToPILImage()])
    
    videoFrames = dataloader.video(root=extractionPath, transform=transform)
    videoFramesloader = torch.utils.data.DataLoader(videoFrames, batch_size = args.batch_size, shuffle=False)

    flowComp = model.UNET(6,4)
    flowComp.to(device)
    for param in flowComp.parameters():
        param.requires_grad = False # Do not update

    ArbTimeFlowIntrp = model.UNET(20, 5)
    ArbTimeFlowIntrp.to(device)
    for param in ArbTimeFlowIntrp.parameters():
        param.requires_grad = False

    flowBackWarp = model.backwarp(videoFrames.dim[0], videoFrames.dim[1], device)
    flowBackWarp.to(device)
    # Load our pre trained model/checkpoint
    dict1 = torch.load(args.checkpoint, map_location='cpu')
    ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
    flowComp.load_state_dict(dict1['state_dictFC'])
    
    frameCounter = 1
    with torch.no_grad():
        for _, (frame0, frame1) in enumerate(tqdm(videoFramesloader), 0):
            I0 = frame0.to(device)
            I1 = frame1.to(device)

            flowOut = flowComp(torch.cat(I0, I1), dim=1)
            # Extract flows between current frame and next frame (F_0_1, F_1_0)
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]
            for batchIndex in range(args.batch_size):
                (TP(frame0[batchIndex].detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex)+".jpg"))
            frameCounter +=1

            for intermediateIndex in range(1, args.sf):
                t = float(intermediateIndex)/args.sf
                temp = -t * (1-t)
                fCoeff = [temp, t*t, (1-t)*(1-t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                g_IO_F_t_0 = trainFlowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = trainFlowBackWarp(I1, F_t_1)

                intrpOut = ArbTimeFlowInter(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_IO_F_t_0),dim=1))

                F_t_0_f = intrpOut[:, :2, :, :]+ F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :]+ F_t_1
                V_t_0 = F.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1 = 1-V_t_0
                # Get the intermediate frames and intermediate flows
                g_IO_F_t_0_f = trainFlowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = trainFlowBackWarp(I1, F_t_1_f)
                wCoeff = [1-t, t]
                Ft_p = (wCoeff[0] * V_t_0 * g_IO_F_t_0_f + wCoeff[1] *V_t_1 * g_I1_F_t_1_f)/(wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
                
                for batchIndex in range(args.batch_size):
                    (TP(Ft_p[batchIndex].cpu().detach())).resize(videoFrames.origDim, Image.BILINEAR).save(os.path.join(outputPath, str(frameCounter + args.sf * batchIndex) + ".jpg"))
                frameCounter+=1
            frameCounter += args.sf *(args.batch_size - 1)
        
    create_video(outputPath)

    rmtree(extractionDir)

    exit(0)
main()