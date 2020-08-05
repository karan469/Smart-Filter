import cv2
import urllib
import numpy as np
from PIL import Image
from io import BytesIO
import itertools
import torch, torchvision

from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
# from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import  GenericMask
from detectron2.data import MetadataCatalog
# import detectron2
from detectron2.config import get_cfg

class detectron(object):
	"""docstring for detectron"""
	def __init__(self):
		super(detectron, self).__init__()
		self.modelname = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
		# self.modelname = "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"
		self.predictor = None
		self.SCORE_THRESHOLD = 0.7
		self.AREA_FRACTION_THRESHOLD = 0.1

		self.cfg = get_cfg()
		self.cfg.MODEL.DEVICE='cpu'
		self.model = build_model(self.cfg)

		self.cfg.merge_from_file(model_zoo.get_config_file(self.modelname))
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
		self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.modelname)
		self.predictor = DefaultPredictor(self.cfg)

		print('Predictor loaded...')

	def union_masks(self, im1, im2):
		assert(im1.shape == im2.shape)
		return im1 | im2

	def return_attributes(self, path):
		im = cv2.imread(path)
		output = self.predictor(im)
		print('STATUS: Detectron2 job done.')

		self.SCORE_THRESHOLD = 0.7
		AREA_FRACTION_THRESHOLD = 0.1

		# storing attributes
		predictions = output["instances"].to("cpu")
		scores = predictions.scores if predictions.has("scores") else None
		classes = predictions.pred_classes if predictions.has("pred_classes") else None

		if predictions.has("pred_masks"):
				masks = [GenericMask(x, im.shape[0], im.shape[1]) for x in np.asarray(predictions.pred_masks)]
		else:
			masks = None

		humans = (classes==0).nonzero().reshape(-1).numpy()
		if(humans.shape[0]==0):
			return None
		temp = []
		for i in range(len(humans)):
			if(scores[humans[i]]>self.SCORE_THRESHOLD and masks[i].area()/(im.shape[0] * im.shape[1])>self.AREA_FRACTION_THRESHOLD):
			# if(True):
				temp.append(humans[i])

		humans = temp
		human_masks = []

		for i in range(len(masks)):
			if(i in humans):
				human_masks.append(masks[i])

		return im, human_masks

	# boxes_array => a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0 are the coordinates of the image's top left corner. x1 and y1 are the coordinates of the image's bottom right corner.
	def bound_human_boxes(self, boxes):
		boxes_array = boxes.tensor.numpy()
		human_crop_images = []
		for i in boxes_array:
			x0, y0, x1, y1 = [int(x) for x in i]
			width = x1 - x0
			height = y1 - y0
			human_crop_images.append(im[y0:y1, x0:x1, :])

		return human_crop_images

	def return_union_mask(self, im, human_masks):
		union_mask = human_masks[0].mask
		if len(human_masks)>1:
			for i in range(1, len(human_masks)):
				union_mask = self.union_masks(union_mask, human_masks[i].mask)
		return union_mask

	def remove_bg(self, im, union_mask):
		temp = im
		h1_mask = union_mask
		for i in range(h1_mask.shape[0]):
			for j in range(h1_mask.shape[1]):
				if(h1_mask[i][j]==0):
					temp[i][j][:] = 255
		return temp
