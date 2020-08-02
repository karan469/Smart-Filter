print('C 1')
from detectron import detectron
from facedetector import facedetector
import utils

import random, string
import cv2

def addbackground(filename, params):
    DetectronObj = detectron()
    # utils.createThumbnail('snowme.jpg')
    new_image = utils.feature_2(DetectronObj, filename, params)
    temp = filename.split('/')
    temp[-1] = '_'.join(params.split(','))+'_'+random_key(5)+'_'+temp[-1]
    new_filename = '/'.join(temp)
    # cv2.imwrite(new_filename, new_image)
    return new_image

def getPrediction(filename):
    DetectronObj = detectron()
    new_image = utils.feature_1(DetectronObj, filename)
    return new_image
    # return addbackground(filename, 'texture')

# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.applications.vgg16 import decode_predictions
# from tensorflow.keras.applications.vgg16 import VGG16

# def getPrediction(filename):

#     model = VGG16()
#     image = load_img('./uploads/'+filename, target_size=(224, 224))
#     image = img_to_array(image)
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     image = preprocess_input(image)
#     yhat = model.predict(image)
#     label = decode_predictions(yhat)
#     label = label[0][0]
#     print('%s (%.2f%%)' % (label[1], label[2]*100))
#     return label[1], label[2]*100
# # print(getPrediction('image.jpg'))
