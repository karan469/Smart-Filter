print('C 1')
from detectron import detectron
from facedetector import facedetector
import utils

import random, string
import cv2

def addNewbackground(filename, params):
    DetectronObj = detectron()
    new_image = utils.feature_2(DetectronObj, filename, params)
    temp = filename.split('/')
    temp[-1] = '_'.join(params.split(','))+'_'+random_key(5)+'_'+temp[-1]
    new_filename = '/'.join(temp)
    return new_image

def get_feature_1(model, filename):
    # DetectronObj = detectron()
    new_image = utils.feature_1(model, filename)
    return new_image

def get_feature_2(detector, smile_detector, filename, bg_filename, category):
    new_image = utils.feature_2(detector, smile_detector, filename, bg_filename, category)
    if(new_image is None):
        return None
    return new_image
