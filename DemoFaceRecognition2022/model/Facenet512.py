from deepface.basemodels import Facenet
from pathlib import Path
import os
import gdown

from deepface.commons import functions

def loadModel(url = 'facenet512_weights.h5'):

    model = Facenet.InceptionResNetV2(dimension = 512)
    model.load_weights(url)
    return model
