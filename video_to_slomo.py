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
    print('{} -i {} -vsync 0 -qscale:v 2 {}/%06d.jpg'.format(os.path.join(args.ffmpeg_dir, 'ffmpeg'), video, outDir))
    retn = os.system('{} -i {} -vsync 0 -qscale:v 2 {}/%06d.jpg'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), video, outDir))
    if retn:
        error = "Error converting file {}. Exiting".format(video)
    return error
def create_video(dir):
    error = ''
    print('{} -r {} -i {}/%d.jpg =qscale:v 2 {}'.format(os.path.join(args.ffmpeg_dir, 'ffmpeg'), args.fps, dir, args.output))
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
    os.mkdir(extractionPath)
    os.makedir(outputPath)
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
    # Lines 128