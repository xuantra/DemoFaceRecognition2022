import pickle
import time
import cv2
import numpy
from deepface.commons import functions
from deepface.detectors import FaceDetector
from deepface.basemodels import Facenet
from deepface import DeepFace
from numpy import expand_dims
model = Facenet.loadModel()

def predict(img_path, svm_clf=None, threshold=0.6):
    if svm_clf is None and model_path is None:
        raise Exception("Must supply svm classifier either thourgh knn_clf or model_path")
    if svm_clf is None:
        with open(model_path, 'rb') as f:
            svm_clf = pickle.load(f)

    # Find encodings for faces in the test iamge
    face_embedding=DeepFace.represent(img_path,model=model,enforce_detection=False)


model_path="model/svm_model.pickle"
label_encoder_path = 'model/svm_labelEncoder.pickle'
with open(model_path, 'rb') as f:
    svm_clf = pickle.load(f)
with open(label_encoder_path, 'rb') as f:
    lb_encoder = pickle.load(f)

# backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
detector_backend = 'mtcnn'
face_detector = FaceDetector.build_model(detector_backend)
cap = cv2.VideoCapture(0)
freeze = False
face_detected = False
face_included_frames = 0
freezed_frame = 0
tic = time.time()
frame_threshold = 3
time_threshold = 2


while cap.isOpened():
    ret, img = cap.read()
    if img is None:
        break
    raw_img = img.copy()
    resolution = img.shape
    resolution_x = img.shape[1]
    resolution_y = img.shape[0]
    if not freeze:
        try:
            faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align=False)
        except:
            faces = []

        if len(faces) == 0:
            face_included_frames = 0
    else:
        faces = []
    detected_faces = []
    face_index = 0
    for face, (x, y, w, h) in faces:
        if w > 130:  # kich thuoc nhan dang
            face_detected = True
            if face_index == 0:
                face_included_frames = face_included_frames + 1  # tang frame nhan dang khuon mat

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
            if frame_threshold - face_included_frames>0:
                cv2.putText(img, str(frame_threshold - face_included_frames), (int(x + w / 4), int(y + h / 1.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 2)


            detected_faces.append((x, y, w, h))
            face_index = face_index + 1

        # -------------------------------------

    if face_detected and face_included_frames == frame_threshold:
        base_img = raw_img.copy()
        detected_faces_final = detected_faces.copy()
        for detected_face in detected_faces_final:
            x = detected_face[0]
            y = detected_face[1]
            w = detected_face[2]
            h = detected_face[3]
            custom_face = base_img[y:y + h, x:x + w]
            imgprocess = custom_face = functions.preprocess_face(img=custom_face,
                                                                target_size=(160, 160),
                                                                enforce_detection=False, detector_backend='mtcnn')
            img_representation = model.predict(imgprocess)[0, :]
            samples = expand_dims(img_representation, axis=0)
            yhat_pro = svm_clf.predict_proba(samples)
            j = numpy.argmax(yhat_pro)
            name = lb_encoder.classes_[j]
            confidence = yhat_pro[0][j]*100
            # Convert it to a native python variable (str)
            name = name.item()

            text = "{} : {:.2f} %".format(name, confidence)
            cv2.putText(img, text, (int(x + w / 4), int(y + h / 1.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        face_detected = False
        face_included_frames = 0

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
