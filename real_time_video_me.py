import cv2
import imutils
import numpy as np
from PyQt5 import QtGui, QtWidgets
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from load_and_process import preprocess_input

# use the default model
default_emotion_model_path = 'models/emotion_models/_mini_XCEPTION.102-0.66.hdf5'


class Emotion_Rec:
    def __init__(self, model_path=None):
        # use the opensource face detection model
        detection_model_path = 'models/face_models/haarcascade_files/haarcascade_frontalface_default.xml'

        if model_path is None:  # to choose model if not use default
            emotion_model_path = default_emotion_model_path
        else:
            emotion_model_path = model_path

        # face detection use opencv
        self.face_detection = cv2.CascadeClassifier(detection_model_path)

        # load our trained model
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        # labels
        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
                         "neutral"]

    def run(self, frame_in, label_face):
        # frame_in: Camera image
        # label_face: The label object used for the face display screen

        frame = imutils.resize(frame_in, width=300)  # resize the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # transfer to gray image

        # use the model to detect faces
        faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1,
                                                     minNeighbors=5, minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
        preds = []  # predict probability
        label = None  # predict label
        (fX, fY, fW, fH) = None, None, None, None  # position of faces
        frameClone = frame.copy()  # copy the image

        if len(faces) > 0:
            # sorted by the size of faces
            faces = sorted(faces, reverse=False, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

            for i in range(len(faces)):
                (fX, fY, fW, fH) = faces[i]

                # get ROI from faces
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, self.emotion_classifier.input_shape[1:3])
                roi = preprocess_input(roi)
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # predict the probability
                preds = self.emotion_classifier.predict(roi)[0]
                label = self.EMOTIONS[preds.argmax()]  # choose the max probability as result to show

                # draw a rectangle for faces and show the result
                cv2.putText(frameClone, label, (fX, fY - 10),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 0, 255), 1)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 1)

        frameClone = cv2.resize(frameClone, (420, 280))

        # show faces in Qt
        show = cv2.cvtColor(frameClone, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        label_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
        QtWidgets.QApplication.processEvents()

        return zip(self.EMOTIONS, preds), label
