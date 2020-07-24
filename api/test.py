print('C 1')
from detectron import detectron
from facedetector import facedetector
import utils

import random
import cv2

##################### FUNCTION CALLING #####################

#-------------------- DETECTRON2 --------------------#
print('C 2')

# DetectronObj = detectron()

# createThumbnail('snowme.jpg')
# new_image = utils.feature_1(DetectronObj, './snowme.png')
# cv2.imwrite('./x_snowme.png', new_image)

print('C 3')

#-------------------- DETECTRON2 - Face detection --------------------#
print('C 4')

# Face = facedetector(modelname='COCO-Detection/retinanet_R_101_FPN_3x.yaml', filename='./model_final.pth')
# im = cv2.imread('./snowme.png', cv2.IMREAD_COLOR)

# pred_boxes = Face.return_pred_boxes(im)
# print(pred_boxes)

print('C 5')

#-------------------- Background addition --------------------#
DetectronObj = detectron()
# utils.createThumbnail('me_winter.jpg')
new_image = utils.feature_2(DetectronObj, './snowme.png', 'leaves,green')
cv2.imwrite('./x_back_snowme_1.png', new_image)