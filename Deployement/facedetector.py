import numpy as np

from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import  GenericMask
from detectron2.data import MetadataCatalog
import detectron2
from detectron2.config import get_cfg

class facedetector(object):
	"""docstring for facedetector"""
	def __init__(self, modelname, filename):
		super(facedetector, self).__init__()
		self._modelname = modelname
		self._cfg = get_cfg()
		self._cfg.merge_from_file(model_zoo.get_config_file(self._modelname))
		self._cfg.MODEL.WEIGHTS = filename
		self._cfg.MODEL.DEVICE = 'cpu'
		self._cfg.MODEL.RETINANET.NUM_CLASSES = 1
		self._cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.50

		self._predictor = DefaultPredictor(self._cfg)

	def return_pred_boxes(self, im):
		output = self._predictor(im)
		return np.array(output['instances']._fields['pred_boxes'].tensor.cpu(), dtype='int32')
