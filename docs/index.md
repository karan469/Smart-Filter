# Smart Filter

This repository aims to create smart filters, majorly for selfies.

Aim of this project is to create **end-to-end pipeline** from engineering machine learning model to deployement of the same. This is achieved using *Docker* and *Flask* as microservice to host the learned model on an EC2 instance. Seperate models are trained on [COCOdataset](https://cocodataset.org/) using detectron2 on pytorch and other datasets using *tensorflow-gpu.*

**UPDATE (01-08-20): Smile Detection is now backed on ResNet50
architecture! The model is more robust and accurate than ever.**

**UPDATE (25-07-20): Released the code as of now. Web app will be
deployed soon!**

**THE REPOSITORY IS PRIVATE AS OF NOW. ONCE I DEPLOY THE PROJECT ON A
WEB/MOBILE APP USING AWS EC2, I WILL RELEASE THE SOURCE CODE.**

## Overview

Smart filter uses machine **learning to curate image specific filters and background**. There will be little to no input of user in deciding the final image. Smart filter is a useful tool to interact and share pictures across social media. 100s of thousands of images are uploaded every single minute. These picures take many forms such as stories, posts, comments, stickers and more.

There are 2 parts to this project:
- Semantic Segmentation and face detection using [Detectron2 (Facebook AI)](https://github.com/facebookresearch/detectron2)
- Facial features detection (smile as of now).

## Requirements

1.  detectron2
2.  pytorch
3.  cuda 10.1
4.  tensorflow
5.  torchvision

## Examples

### Background Removal

This API can be used to remove background and place either a fixed
preset, or another aesthetically good looking background using
parameters given by user.
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
![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/results/15.jpg)  |  ![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/results/16.jpg)

### Face detection using transfer learning on detectron2

This model is created using 500 images with 1100 instances on faces in the [dataset](https://www.kaggle.com/dataturks/face-detection-in-images)
Thus, the learning is not as rigorous as it could be. But I have achieved an **F1 score of 0.702** which was good enough.
> Insert model quality metrics.
```python
    def return_face_detection_predictor(filename):
        from detectron2.config import get_cfg

        MODEL_PATH = 'COCO-Detection/retinanet_R_101_FPN_3x.yaml'
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))
        cfg.MODEL.WEIGHTS = filename
        cfg.MODEL.RETINANET.NUM_CLASSES = 1
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.50

        return DefaultPredictor(cfg)

    face_predictor = return_face_detection_predictor('/content/drive/My Drive/Colab Notebooks/checkpoint_face.pth')
    face_detection_output = face_predictor(im)
    pred_boxes = np.array(face_detection_output['instances']._fields['pred_boxes'].tensor.cpu(), dtype='int32')
```

Image 1             |  Image 2
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/results/5.png)  |  ![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/results/6.png)

Right now, Face Filter is used primarily for selfies thus, the achieved results are satisfactory.


### Smile Detection

Selfies taken are inherently full of features which can be used to
automate creating appropriate face filters. For example, facial features
can be used as a good way to describe a selfie. I have created only
smile detection so far.\
 The final model predicts on a given face. You can use either a Haar
features based frontal face cascade or another pretrained CNN model to
localise the faces in image.

> Insert model quality metrics.
```python
    smile_predictor = SmileModel('/content/drive/My Drive/Colab Notebooks/smiledetection.h5')
    temp = cv2.resize(temp, dsize=(64, 64))
    temp = temp.reshape(1, 64, 64, 3)
    probability = smile_predictor.predict(temp)
```
Image 1                    |        Image 2            |         Image 3
:-------------------------:|:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/results/2.png)  |  ![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/results/1.png) | ![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/results/3.png)

[](#smart-filters-examples)Smart Filters examples

Input           |  Output with custom background
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/results/9.png) | ![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/results/11.png)

Input           |  Output
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/results/4.png) | ![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/results/8.png)

Input          |  Output
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/results/13.png) | ![](https://raw.githubusercontent.com/karan469/Smart-Filter/master/results/7.png)

Last image is created by giving 'rainbow,texture' as parameters to background selection.
> Insert code snippet.

## I'd ❤️ suggestions ..

1.  How to deploy a deep learning model on platforms such as heroku or aws where the slug size of my docker container is 582 MB as of now, whereas heroku allows a maximum slug size of 500 MB?
2.  Point me in the direction of a good dataset for human facial
    features detection, and face detection (Bigger the better).
    Currently, I am using transfer learning for face detection, on a
    small dataset of 500 images and 1100 faces only, which is giving
    only satisfactory results.
3.  There are some big possible improvements that could be done, such as
    multi-task learning. This might reduce latency in providing results,
    by a huge margin. Let me know your thoughts on that!
