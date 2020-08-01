'''
	Note: This api requires keras==2.3.1. In case this version or whole keras get depreciated,
	please update the api to tensorflow.keras
'''
from facedetector import facedetector
# import tensorflow as tf
import keras
import cv2
import numpy as np

class smiledetector(object):
	"""docstring for smiledetector"""
	def __init__(self, arg):
		super(smiledetector, self).__init__()
		self.model = None
		self.weights_filepath = arg

	def _load_model(self):
		model = keras.applications.ResNet50(
			include_top=False,
			weights=None,
			input_tensor=None,
			input_shape=(75, 75, 3),
			pooling=None,
			classes=2,
		)
		for layer in model.layers:
			layer.trainable = True
		x = keras.layers.Flatten()(model.output)
		x = keras.layers.Dense(256, activation='relu')(x)
		x = keras.layers.Dense(32, activation='relu')(x)

		x = keras.layers.Dense(2, activation='softmax')(x)

		model = keras.models.Model(model.input, x)

		model.load_weights(self.weights_filepath)

		model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

		self.model = model

	def _return_smile_prob(self, im):
		if(self.model==None):
			self._load_model()

		im = cv2.resize(im, (75, 75))
		im = im/255.
		res = self.model.predict(np.array([im]))
		return res

	def _infer_image(self, filename):
		if(self.model==None):
			self._load_model()

		im = cv2.imread(filename)
		im = cv2.resize(im, (75, 75))
		im = im/255.
		res = self.model.predict(np.array([im]))
		if(res[0][0]<0.5):
			return ('Smiling: '+ str(res[0][1]))
		else:
			return ('Not Smiling: '+ str(res[0][0]))

	def _infer_image_batch(self, batch):
		if(self.model==None):
			self._load_model()

		res = self.model.predict(batch)

		return res
		
if __name__ == '__main__':
	detector = smiledetector('../../resnet50_smiledetection.h5')
	print(detector._infer_image('./test1.png'))