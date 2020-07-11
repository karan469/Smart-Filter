# Smart-Filter
This repository aims to create a mobile image segmentation model and incorporate background according to the human emotion

Aim of this project is to create end-to-end pipeline from engineering machine learning model to deployement of the same. This is achieved using Docker and Flask as microservice to host the learned model on an EC2 instance. Seperate models are trained on [COCOdataset](https://cocodataset.org/) using detectron2 on pytorch and other datasets using tensorflow-gpu==2.2.0.

## Overview

Smart filter uses machine learning to curate image specific filters and background. There will be little to no input of user in deciding the final image.
Smart filter is a useful tool to interact and share pictures across social media. 100s of thousands of images are uploaded every single minute. These picures take many forms such as stories, posts, comments, stickers and more.

There are 3 parts to this project:
- Semantic Segmentation using [detectron2](https://github.com/facebookresearch/detectron2)
- Face detection using transfer learning on detectron2 itself.
- Smile detection on detected face.

## Pre-requisites
1. pytorch
2. cuda 10.1
3. tensorflow
4. torchvision
5. numpy
6. detectron2
7. PIL

## Examples

### Background Removal

> This API can be used to remove background and place either a fixed preset, or another aesthetically good looking background using parameters given by user.
> Insert code snippet here.

Original Image             |  Background removal using Semantic Segmentation
:-------------------------:|:-------------------------:
![](https://github.com/CRekkaran/Smart-Filter/blob/master/Semantic%20Segmentation/person_selfie.jpg)  |  ![](https://github.com/CRekkaran/Smart-Filter/blob/master/Semantic%20Segmentation/unsplashFilter.jpg)

### Face detection using transfer learning on detectron2

> This model is created using 510 images with 1600 instances on faces in the [dataset](https://www.kaggle.com/dataturks/face-detection-in-images)
> Thus, the learning is not as rigorous as it could be. But I have achieved an F1 score of 0.702 which was good enough.
> Insert model quality metrics.
> Insert code snippet here.

Image 1             |  Image 2
:-------------------------:|:-------------------------:
![](https://github.com/CRekkaran/Smart-Filter/blob/master/Face%20detection%20using%20Detectron2/index.png)  |  ![](https://github.com/CRekkaran/Smart-Filter/blob/master/Face%20detection%20using%20Detectron2/index1.png)

Right now, Face Filter is used primarily for selfies thus, the achived results are satisfactory.

### Smile Detection

> Selfies taken are inherently full of features which can be used to automate creating appropriate face filters. For example, facial features can be used as a good way to describe a selfie. I have created only smile detection so far.
> Insert model quality metrics.
> Insert code snippet here.

Image 1             |  Image 2 | Image 3
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/CRekkaran/Smart-Filter/blob/master/Smile%20Detection/1.png)  |  ![](https://github.com/CRekkaran/Smart-Filter/blob/master/Smile%20Detection/2.png) | ![](https://github.com/CRekkaran/Smart-Filter/blob/master/Smile%20Detection/not%20smiling.png)

## Smart Filters examples

Example 1             |  Example 2
:-------------------------:|:-------------------------:
![](https://github.com/CRekkaran/Smart-Filter/blob/master/Semantic%20Segmentation/supportpride.png) | ![](https://github.com/CRekkaran/Smart-Filter/blob/master/Final%20Ensemble/index3_low_res.png)

Example 1 is created by giving 'rainbow,texture' as parameters to background selection.
> Insert code snippet.
