# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model

# load model
model = load_model('./smiledetection.hdf5')
# summarize model.
model.summary()