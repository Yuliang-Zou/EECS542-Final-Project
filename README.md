# EECS542-Final-Project
This repository contains the source code to repoduce our guided method results on [DAVIS Challenge 2016](http://davischallenge.org/index.html). For the unguided method, please check [this repository](https://github.com/gaochen315/DAVIS-Challenge---RPCA) for further details.

<table><tr><td>
<img src='./demo/cows.gif'>
<img src='./demo/drift_chicane.gif'>
</td>

## Requirements

- Torch

- PyTorch

- [Optional] TensorFlow r0.11

## Dataset Preparation

1. `mkdir data`

2. Download DAVIS data [here](http://davischallenge.org/code.html), then put it under the `data/` directory.

3. Download SegTrack v2 data [here](http://web.engr.oregonstate.edu/~lif/SegTrack2/dataset.html), then put it under the `data/` directory.

4. Download YouTubeObject data [here](http://vision.cs.utexas.edu/projects/videoseg/data_download_register.html), then put it under the `data/` directory.

5. [Optional] Download my augmented data [here](https://drive.google.com/file/d/0B2SnTpv8L4iLaHpobThSNnNiMkU/view?usp=sharing), then put it under the `data/` directory. You can also modify the `data_augment.py` file to generate your own dataset. 

## Download Pre-trained model

1. Faster R-CNN VGG16 (.h5): Please download it [here](https://drive.google.com/open?id=0B4pXCfnYmG1WOXdpYVFybWxiZFE), then put it under `./faster_rcnn_pytorch/model/`.

2. SharpMask model (.t7): `wget https://s3.amazonaws.com/deepmask/models/sharpmask/model.t7`. Or you can also download it [here](https://drive.google.com/file/d/0B2SnTpv8L4iLUW1EVmhwVFp2ZFU/view?usp=sharing). Then, put it under `sharpMask/pretrained/`

## How to run (Suppose you have downloaded the full dataset and models)

1. `cd faster_rcnn_pytorch/; python tracking` Then you can find the cropped images under `demo` folder, and a `crop.npy` file in this folder.

2. `cd ../sharpMask/; th computeProposals.lua -model ./pretrained -path ../faster_rcnn_pytorch/demo/ -dump ./output/` Then you can find segmented images under your specified directory. 

3. Move the previous `crop.npy` file to the same folder of the segmented images. Then run `return_box.py` under the `sharpMask` folder (you might want to specify the output path), you will get the final mask results.

## Evaluate

Please check their [official website](http://davischallenge.org/index.html) for reference. They change the evaluation metric in 2017, so you might not use the lastest eval code directly.

## Misc

We have also tried some other methods, but we did not get convicing results from such methods. We believe that they can be adapted and improved (if you have enough time the train the test). If you find it useful for you, feel free to use our code.

## Credit

We adapted the code from a [PyTorch implementation](https://github.com/longcw/faster_rcnn_pytorch) of [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf), and the official [Torch implementation](https://github.com/facebookresearch/deepmask) of [Learning to Refine Object Segments](https://arxiv.org/pdf/1603.08695.pdf). If you use this part of our code, please cite the relevant papers.