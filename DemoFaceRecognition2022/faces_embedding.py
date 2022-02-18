from deepface import DeepFace
from imutils import paths
import numpy as np
import argparse
import pickle
import cv2
import os
from model import Facenet128

dataset_path= 'dataset'
model_path='model'
model_path=os.path.join(model_path,'facenet_weights.h5')
if not os.path.isfile(model_path):
    print('file weights not exist')
model = Facenet128.loadModel(model_path)

knownEmbeddings = []
knownNames = []

for name in os.listdir(dataset_path):
    file_path=os.path.join(dataset_path,name)
    for file in paths.list_files(file_path):
        face_embedding=DeepFace.represent(file,detector_backend='mtcnn',model=model,enforce_detection=False)
        knownNames.append(name)
        knownEmbeddings.append(face_embedding)

# save to output
data = {"embeddings": knownEmbeddings, "names": knownNames}
with open('embedding.pickle', "wb") as write_file:
    pickle.dump(data,write_file)


f = open('embedding.pickle',"rb")
data = pickle.load(f)
print(len(data["embeddings"]))
f.close()