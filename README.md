# Super_SloMo_Implementation

## Introduction

This is based on the paper: Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video Interpolation by Huaizu Jiang, Deqing Sun, Varun Jampani, Ming-Hsuan Yang, Erik Learned-Miller, Jan Kautz [paper](https://arxiv.org/abs/1712.00080)

## Original Source

The code is a fork on the [Super SloMo Implementation](https://github.com/avinashpaliwal/Super-SloMo/blob/master/data/create_dataset.py) by [Avinash Paliwal](https://github.com/avinashpaliwal).

![Original_Video_CheckPoint_Pretrained](https://media.giphy.com/media/dxgxdqZfsT8nOJtWou/giphy.gif)
![Pretrained_Video_CheckPoint_Pretrained](https://media.giphy.com/media/YrqHjJmlwLrx3hkl4G/giphy.gif)

## Google Colab Training

I created a complimentary notebook for this so that I can train this on Colab. Google Colab is a great free resource that allows us to make use of a free GPU.

## Changes from the Original Repository

I did some minimal changes in the code that is different from the original. I have added *Cosine Annealing* to the learning rate with restarts. The idea I just want to check how much the losses improve and the convergence.

```text
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,10*len(trainloader))
```

and for the restarts:

```text
if (epoch%10 == 0):
    print('Restarting the LR')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,10*len(trainloader))
```

## Results

The sample video for this is a 720p version of Big Buck Bunny which is 1Mb in size which is [downloadable here](https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4). The original running time is `time=00:00:05.28` and the total detected frames by ffmpeg is `132`.

I used the trained [checkpoint](https://drive.google.com/open?id=1ydwy8XNGkZALpY2dpeN3q2YQKorLs7LS) for the model in 50 epochs under the smaller dataset. The scale factor was set to 4 and the output fps was set to 60fps. After processing I got a file size of 8Mb, a total time of `time=00:00:08.36` and `505` frames created.

Some noticeable issues can be seen in the resulting video but given that this was only trained for 50 epochs and in a reduced dataset I think the fact that it can produce the intermediate frames is already amazing.

Some key points to be improved are the stutter in the overall frame. If we focus on the grass near the left side of the screen we can see some shaking in the frame that was not in the original. Also, notice the sort of mirage that was left behind when the bunny was raising his right hand to stretch. We can clearly see the sort of wave-like blob in the grass where his arms passed through. It is very much an open project and there is a lot more to do if this was pursued further.

Here are the resulting GIF of the model after 50 epochs on a small subset.

![SuperSloMo-330-BigBuckBunny](https://raw.githubusercontent.com/iocfinc/Super_SloMo_Implementation/master/GIF/SuperSloMo-BBB-330.gif)

## Possible Future

Future work in this would be figuring out how to apply Random Pixel Shuffle to the upsampling of the intermediate images. Right now the options are either Bilinear or LANCZOS. Also, it would be good to train the model in a larger dataset for longer. The baseline for the original repo model was 200 epochs for the Adobe240fps dataset. I did a trial run using the same dataset and 1 epoch was taking almost 1 hour to train.