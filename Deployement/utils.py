import random
import cv2
from PIL import Image
import requests
from io import BytesIO
import numpy as np

def addPreset(im, union_mask):
	green = random.choice([40, 100, 255, 240, 200, 80])
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
	im, human_masks = detector.return_attributes(filename)

	print('STATUS: return_attributes done.')

	final_mask = detector.return_union_mask(im, human_masks)
	im = addPreset(im, final_mask)
	print('STATUS: addPreset done.')
	return im

def feature_2(detector, smile_detector, filename, bg_filename, category):
	'''
		Add unsplash image as background.
		TO-DO: run detector and loadImage in parallel (using threads or processes)
	'''
	packed_segment = detector.return_attributes(filename)
	if(packed_segment is None):
		return None
	im, human_masks = packed_segment
	final_mask = detector.return_union_mask(im, human_masks)
	print('STATUS: Segment mask calculated')

	bg = None
	if(bg_filename):
		bg = cv2.imread(bg_filename)
	elif(category is None):
		pass
	else:
		bg_resolution_string = str(im.shape[1])+'x'+str(im.shape[0])
		bg = loadImage('https://source.unsplash.com/'+bg_resolution_string+'/?'+category)

	print('STATUS: Background Image loaded')

	h,w = final_mask.shape[0],final_mask.shape[1]

	WHITE_BORDER_FRACTION = 0.07

	if(bg is not None):
		bg = bg[:im.shape[0], -im.shape[1]:, :] # second check: no need tbh

		# bg = bg[int(h*WHITE_BORDER_FRACTION):int((1-WHITE_BORDER_FRACTION)*h), int(h*WHITE_BORDER_FRACTION):int((1-WHITE_BORDER_FRACTION)*h),:]
		# bg = np.pad(bg, pad_width = int(h*(1-WHITE_BORDER_FRACTION)))

		print('im.shape = ', im.shape)
		print('bg.shape = ', bg.shape)
		print('final_mask.shape = ', final_mask.shape)

		im[:,:,0] = np.where(final_mask!=0, im[:,:,0], bg[:,:,0])
		im[:,:,1] = np.where(final_mask!=0, im[:,:,1], bg[:,:,1])
		im[:,:,2] = np.where(final_mask!=0, im[:,:,2], bg[:,:,2])
	else:
		green = random.choice([40, 100, 255, 240, 200, 80])
		for i in range(final_mask.shape[0]):
			for j in range(final_mask.shape[1]):
				if(final_mask[i][j]==0):
					if(i>int(h*WHITE_BORDER_FRACTION) and i<int((1-WHITE_BORDER_FRACTION)*h) and j>int(w*WHITE_BORDER_FRACTION) and j<int((1-WHITE_BORDER_FRACTION)*w)):
						red = 255*(1-(j/final_mask.shape[1]))
						im[i][j] = [red, green, 255]
					else:
						im[i][j] = [255, 255, 255]

	if(smile_detector):
		print('Smile should be detected')
		smiling_prob_final = smile_detector._return_smile_prob(im)[0][1]

		print('Smiling prediction: ', smiling_prob_final)

		isSmiling = smiling_prob_final > 0.5
		if isSmiling:
			textOnImage = 'Smiling: ' + str('%.2f' % smiling_prob_final)
			textOnImage = '#KEEPSMILING ;)'
		else:
			textOnImage = 'Not Smiling: ' + str('%.2f' % (1-smiling_prob_final))
			textOnImage = '#OFFMOOOOD'

		print('STATUS: Smiling probability: ', smiling_prob_final)

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

def feature_3(segment_detector, smile_detector, filename, category):
	'''
		Add smile related caption to the image
	'''
	WHITE_BORDER_FRACTION = 0.07

	im, human_masks, boxes = segment_detector.return_attributes(filename)
	final_mask = segment_detector.return_union_mask(im, human_masks)

	print('Segment mask calculated')

	# bg = loadImage('https://source.unsplash.com/1500x1500/?'+category)

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
