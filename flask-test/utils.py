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
	# with urllib.request.urlopen(URL) as url:
	#	 img = tf.keras.preprocessing.image.load_img(BytesIO(url.read()), target_size=(w, h))
	response = requests.get(URL)
	img = Image.open(BytesIO(response.content))
	return np.array(img)

def feature_1(detector, filename):
	# Remove bg and output image with bg-color-preset
	im, human_masks, boxes = detector.return_attributes(filename)

	# createThumbnail('me_winter.jpg')
	final_mask = detector.return_union_mask(im, human_masks)
	im = addPreset(im, final_mask)
	return im
	# cv2.imwrite('./x_'+filename.split('./')[1], im)

def feature_2(detector, filename, catagory):
	'''
		Add unsplash image as background.
		TO-DO: run detector and loadImage in parallel (using threads or processes)
	'''
	im, human_masks, boxes = detector.return_attributes(filename)
	final_mask = detector.return_union_mask(im, human_masks)
	print('Segment mask calculated')
	
	bg = loadImage('https://source.unsplash.com/1080x1920/?'+catagory)
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