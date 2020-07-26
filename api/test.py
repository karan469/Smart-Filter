print('C 1')
from detectron import detectron
from facedetector import facedetector
import utils

import random, string
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
# Returns a random alphanumeric string of length 'length'
def random_key(length):
	return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(length))
	
def addbackground(filename, params):
	DetectronObj = detectron()
	# utils.createThumbnail('snowme.jpg')
	new_image = utils.feature_2(DetectronObj, filename, params)
	temp = filename.split('/')
	temp[-1] = '_'.join(params.split(','))+'_'+random_key(5)+'_'+temp[-1]
	new_filename = '/'.join(temp)
	cv2.imwrite(new_filename, new_image)

addbackground('./snowme.png', 'flowers')