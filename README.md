# UDC-Residual-and-Dense-UNet
UDC Challenge

# Code Framework
The overall code framework mainly consists of four parts - `Config`, `Data`, `Model` and `Network`.

Let us take the train commad `python train.py -opt options/train/train_UDC.yml` for example. A sequence of actions will be done after this command. 

## Table of Contents
1. [Config]
1. [Data]
1. [Model]
1. [Network]
1. [Train/Test]
1. [Utils]

## Config
#### [`options/`] Configure the options for data loader, network structure, model, training strategies and etc.

- `yml` file is used to configure options and [`options/options.py`] will convert the json file to python dict.
- `yml` file uses `null` for `None`; and supports `//` comments, i.e., in each line, contents after the `//` will be ignored. 
- Supports `debug` mode, i.e, model name start with `debug_` will trigger the debug mode.
- The configuration file and descriptions can be found in [`options`].

## Data
#### [`data/`] A data loader to provide data for training, validation and testing.

- A separate data loader module. You can modify/create data loader to meet your own needs.
- Uses `cv2` package to do image processing, which provides rich operations.
- Supports reading files from image folder or `lmdb` file. For faster IO during training, recommand to create `lmdb` dataset first. More details including lmdb format, creation and usage can be found in our [lmdb wiki].
- [`data/util.py`] provides useful tools. For example, the `MATLAB bicubic` operation; rgb<-->ycbcr as MATLAB.
- Now, we convert the images to format NCHW, [0,1], RGB, torch float tensor.

## Model
#### [`models/`] Construct different models for training and testing.

- A model mainly consists of two parts - [network structure] and [model defination, e.g., loss definition, optimization and etc]. The network description is in the [Network part].
- Based on the [`base_model.py`], we define different models, e.g., [`SR_model.py`], .

## Network
#### [`models/modules/`]Construct different network architectures.

- The network is constructed in [`models/network.py`] and the detailed structures are in [`models/modules`].

## Train/Test
Please first check the .yml file, the location of the data set meets the requirements.
#### Train
- cd codes
- python train.py -opt options/train/train_UDC.yml
#### Test
- Download link of trained model: https://pan.baidu.com/s/1KTpGej6Le6lILrZD3CLiIQ  (PW: 6ije) (TOLED)
  or Download link of trained model: https://pan.baidu.com/s/1wvYINJeWQaGWaTmwnW60JQ (PW: crfq) (POLED)
- cd codes
- python  prep_result.py 
It is worth noting that when testing, you need to put the tested .mat file in ‘codes/toled_test_display.mat’.

## Utils
#### [`utils/`] Provide useful utilities.

- [logger.py] provides logging service during training and testing.
- Support to use [tensorboard] to visualize and compare training loss, validation PSNR and etc. 
- [progress_bar.py] provides a progress bar which can print the progress. 

Authors: Qirui Yang, Yihao Liu, Jigang Tang, Tao Ku [paper]

If you find our work is useful ,please kindly cite it.
'''
@InProceedings{yang2020residual,
author = {Qirui Yang and Yihao Liu and Jigang Tang and Tao Ku},
title = {Residual and Dense UNet for Under-display Camera Restoration},
booktitle = {European Conference on Computer Vision Workshops},
year = {2020},
}
'''
