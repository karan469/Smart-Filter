# Smart Filter

Aim of this project is to create **end-to-end pipeline** from engineering machine learning model to deployement of the same. This is achieved using *Docker* and *Flask* as microservice to host the learned model on an EC2 instance. Seperate models are trained on [COCOdataset](https://cocodataset.org/) using detectron2 on pytorch and other datasets using *tensorflow-gpu==2.2.0.*

## Overview

Smart filter uses machine **learning to curate image specific filters and background**. There will be little to no input of user in deciding the final image.
Smart filter is a useful tool to interact and share pictures across social media. 100s of thousands of images are uploaded every single minute. These picures take many forms such as stories, posts, comments, stickers and more.

There are 3 parts to this project:
- Semantic Segmentation using [detectron2 - Facebook AI](https://github.com/facebookresearch/detectron2)
- Face detection using transfer learning on *detectron2* itself.
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

This API can be used to remove background and place either a fixed preset, or another aesthetically good looking background using parameters given by user.

```python
path = '/content/'
#returns necessary attrbutes and results from model output
im, human_masks, boxes = return_attributes(path+'me.jpg')
print('Masks calculated')

# list of singular human images
final_images = bound_human_boxes(boxes)

# final mask => NxM bool matrix : value is 1 if human is present on the pixel
final_mask = return_union_mask(im, human_masks)
print('Final Mask created')

# Remove background and show the resultant image 
cv2_imshow(remove_bg(im, final_mask))
```

Original Image             |  Background removal using Semantic Segmentation
:-------------------------:|:-------------------------:
![](https://github.com/karan469/Smart-Filter/blob/master/Semantic%20Segmentation/person_selfie.jpg)  |  ![](https://github.com/karan469/Smart-Filter/blob/master/Semantic%20Segmentation/unsplashFilter.jpg)

### Face detection using transfer learning on detectron2

This model is created using 510 images with 1600 instances on faces in the [dataset](https://www.kaggle.com/dataturks/face-detection-in-images)
Thus, the learning is not as rigorous as it could be. But I have achieved an **F1 score of 0.702** which was good enough.
> Insert model quality metrics. Insert code snippet.


Image 1             |  Image 2
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/Face%20detection%20using%20Detectron2/index.png?token=AH47IOSZ4WGFR44JXCTVWZC7CSCBE)  |  ![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/Face%20detection%20using%20Detectron2/index1.png?token=AH47IOSDXDEXVMNJDGP36BC7CSCCS)

Right now, Face Filter is used primarily for selfies thus, the achieved results are satisfactory.

### Smile Detection

Selfies taken are inherently full of features which can be used to automate creating appropriate face filters. For example, facial features can be used as a good way to describe a selfie. I have created only smile detection so far.  
The final model predicts on a given face. You can use either a Haar features based frontal face cascade or another pretrained CNN model to localise the faces in image.
> Insert model quality metrics.

```python
smile_predictor = SmileModel('/content/drive/My Drive/Colab Notebooks/smiledetection.h5')
temp = cv2.resize(temp, dsize=(64, 64))
temp = temp.reshape(1, 64, 64, 3)
probability = smile_predictor.predict(temp)
```

Image 1             |  Image 2 | Image 3
:-------------------------:|:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/Smile%20Detection/1.png?token=AH47IOXHWTKLPEXY7PN6M727CSCEY)  |  ![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/Smile%20Detection/2.png?token=AH47IOXHM3T5GAT5K26THSC7CSCGI) | ![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/Smile%20Detection/not%20smiling.png?token=AH47IOQM7LDSQ67PZQPD5O27CSCIK)

## Smart Filters examples

Example 1             |  Example 2
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/Final%20Ensemble/index3_low_res.png?token=AH47IOVZM4UJZCTIAEEZVW27CSCRC) | ![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/Semantic%20Segmentation/supportpride.png?token=AH47IOSTKOYLZS4OVBCMUCK7CSCO2)

Example 2 is created by giving 'rainbow,texture' as parameters to background selection.
> Insert code snippet.
