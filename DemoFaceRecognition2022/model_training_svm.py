from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

from sklearn.svm import SVC


def train_svm_model(embeddings_path="", classifier_model_path="", label_encoder_path=""):
    # Load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(embeddings_path, "rb").read())

    # Encode the labels
    print("encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])


    print("training model...")
    recognizer = SVC(C=1, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    print("saving model...")
    with open(classifier_model_path, "wb") as write_file:
        pickle.dump(recognizer, write_file)


    with open(label_encoder_path, "wb") as write_file:
        pickle.dump(le, write_file)


if __name__ == "__main__":
    train_svm_model('embedding.pickle', 'model/svm_model.pickle', 'model/svm_labelEncoder.pickle')
