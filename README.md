# Smart-Filter
This repository aims to create a mobile image segmentation model and incorporate background according to the human emotion

Aim of this project is to create end-to-end pipeline from engineering machine learning model to deployement of the same. This is achieved using Docker and Flask as microservice to host the learned model on an EC2 instance. Seperate models are trained on [COCOdataset](https://cocodataset.org/) using detectron2 on pytorch and other datasets using tensorflow-gpu==2.2.0.

# Overview

Smart filter uses machine learning to curate image specific filters and background. There will be little to no input of user in deciding the final image.
Smart filter is a useful tool to interact and share pictures across social media. 100s of thousands of images are uploaded every single minute. These picures take many forms such as stories, posts, comments, stickers and more.

There are 3 parts to this project:
- Semantic Segmentation using [detectron2](https://github.com/facebookresearch/detectron2)
- Face detection using transfer learning on detectron2 itself.
- Smile detection on detected face.

# Pre-requisites
1. pytorch
2. cuda 10.1
3. tensorflow
4. torchvision
5. numpy
6. detectron2
7. PIL
