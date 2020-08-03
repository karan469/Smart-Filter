import random
import cv2
from PIL import Image
import requests
from io import BytesIO
import numpy as np

def addPreset(im, union_mask):
	green = random.choice([40, 100, 240, 200, 80])
	temp = im
	h1_mask = union_mask
	WHITE_BORDER_FRACTION = 0.07
	for i in range(h1_mask.shape[0]):
		for j in range(h1_mask.shape[1]):
			
			if(h1_mask[i][j]==0):
				if(i>int(h1_mask.shape[0]*WHITE_BORDER_FRACTION) and i<int((1-WHITE_BORDER_FRACTION)*h1_mask.shape[0]) and j>int(h1_mask.shape[1]*WHITE_BORDER_FRACTION) and j<int((1-WHITE_BORDER_FRACTION)*h1_mask.shape[1])):
					red = 255*(1-(j/h1_mask.shape[1]))
					temp[i][j] = [red, green, 255]
				else:
					temp[i][j] = [255, 255, 255]
	return temp	

def createThumbnail(filename):
	'''
		createThumbnail('me.jpg')
		creates lower resolution png format image
	'''
	from PIL import Image
	path = './'
	size = 1080, 1296
	im = Image.open(path+filename)
	im.thumbnail(size, Image.ANTIALIAS)
	im.save(path+filename.split('.')[0]+'.png', "PNG")

def loadImage(URL):
	response = requests.get(URL)
	img = Image.open(BytesIO(response.content))
	return np.array(img)

def feature_1(detector, filename):
	# Remove bg and output image with bg-color-preset
	# im, human_masks, boxes = detector.return_attributes(filename)
	im, human_masks = detector.return_attributes(filename)

	print('return_attributes done.')

	final_mask = detector.return_union_mask(im, human_masks)
	im = addPreset(im, final_mask)
	print('addPreset done.')
	return im

def feature_2(detector, filename, catagory):
	'''
		Add unsplash image as background.
		TO-DO: run detector and loadImage in parallel (using threads or processes)
	'''
	im, human_masks, boxes = detector.return_attributes(filename)
	final_mask = detector.return_union_mask(im, human_masks)
	print('Segment mask calculated')
	
	bg = loadImage('https://source.unsplash.com/1500x1500/?'+catagory)
	print('Unsplash image loaded')

	WHITE_BORDER_FRACTION = 0.07
	for i in range(final_mask.shape[0]):
		for j in range(final_mask.shape[1]):
			
			if(final_mask[i][j]==0):
				if(i>int(final_mask.shape[0]*WHITE_BORDER_FRACTION) and i<int((1-WHITE_BORDER_FRACTION)*final_mask.shape[0]) and j>int(final_mask.shape[1]*WHITE_BORDER_FRACTION) and j<int((1-WHITE_BORDER_FRACTION)*final_mask.shape[1])):
					red = 255*(1-(j/final_mask.shape[1]))
					im[i][j] = bg[i,j,:]
				else:
					im[i][j] = [255, 255, 255]
	return im	

def feature_3(segment_detector, face_detector, smile_detector, filename, catagory):
	'''
		Add smile related caption to the image
	'''
	WHITE_BORDER_FRACTION = 0.07

	im, human_masks, boxes = segment_detector.return_attributes(filename)
	final_mask = segment_detector.return_union_mask(im, human_masks)
	
	print('Segment mask calculated')
	
	# bg = loadImage('https://source.unsplash.com/1500x1500/?'+catagory)
	
	bg = cv2.imread('./background.jfif')

	print('Unsplash image loaded')

	############################### ###############################
	print('Removing background and adding preset')
	for i in range(final_mask.shape[0]):
		for j in range(final_mask.shape[1]):
			
			if(final_mask[i][j]==0):
				if(i>int(final_mask.shape[0]*WHITE_BORDER_FRACTION) and i<int((1-WHITE_BORDER_FRACTION)*final_mask.shape[0]) and j>int(final_mask.shape[1]*WHITE_BORDER_FRACTION) and j<int((1-WHITE_BORDER_FRACTION)*final_mask.shape[1])):
					red = 255*(1-(j/final_mask.shape[1]))
					im[i][j] = bg[i,j,:]
				else:
					im[i][j] = [255, 255, 255]

	print(im.shape[1]*WHITE_BORDER_FRACTION)

	smiling_prob_final = smile_detector._return_smile_prob(im)[0][1]

	print('Smiling prediction: ', smiling_prob_final)

	isSmiling = smiling_prob_final > 0.5
	if isSmiling:
		textOnImage = 'Smiling: ' + str('%.2f' % smiling_prob_final)
		textOnImage = '#KEEPSMILING ;)'
	else:
		textOnImage = 'Not Smiling: ' + str('%.2f' % (1-smiling_prob_final))
		textOnImage = '#OFFMOOOOD'
	############################### ###############################

	font				   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (100,700)
	fontScale			  = 3
	fontColor			  = (10,0,250)
	lineType			   = 3

	cv2.putText(
		im,
		textOnImage, 
		bottomLeftCornerOfText, 
		font, 
		fontScale,
		fontColor,
		lineType
	)

	return im