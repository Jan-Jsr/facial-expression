from ui.window import Ui_Form
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QProgressBar, QTextBrowser
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QMovie
from real_time_video_me import Emotion_Rec  # get faces and emotion recognize
from os import getcwd
import numpy as np
import cv2
import time
import sys

sys.path.append('../')
import qrc.resource  # dirName.fileName


class MainWindow(QtWidgets.QMainWindow, Ui_Form):
    # class MainWindow(object):
    def __init__(self, parent=None):
        # super(MainWindow, self).__init__(parent)
        QMainWindow.__init__(self)
        Ui_Form.__init__(self)

        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
                         "neutral"]
        self.path = getcwd()
        self.timer_camera = QtCore.QTimer()  #timer

        self.setupUi(self)
        self.retranslateUi(self)
        self.slot_init()  # Slot function setting

        # Set interface animation
        gif = QMovie(':/images/ui/scan.gif')
        self.label_face.setMovie(gif)
        gif.start()

        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0  # camera
        self.model_path = None  # path of model: use default
        self.emotion_model = None
        self.image = None
        self.showResult()

    def slot_init(self):  # Define slot function
        self.toolButton_model.clicked.connect(self.choose_model)
        self.toolButton_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.toolButton_file.clicked.connect(self.choose_pic)

    def choose_model(self):
        # default setting
        self.timer_camera.stop()
        self.cap.release()
        self.label_face.clear()
        self.textBrowser_result.setText('None')
        self.textBrowser_time.setText('0 s')
        self.textEdit_camera.setText('Camera off')
        self.showResult()

        # choose model file
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,
                                                                "Select a file...", getcwd(),  # file path
                                                                "Model File (*.hdf5)")  # only hdf5 file can be choose
        # default model hits message
        if fileName_choose != '':
            self.model_path = fileName_choose
            self.textEdit_model.setText(fileName_choose)
        else:
            self.textEdit_model.setText('Using default model')

        gif = QMovie(':/images/ui/scan.gif')
        self.label_face.setMovie(gif)
        gif.start()

    def button_open_camera_click(self):
        # check camera status
        if self.timer_camera.isActive() is False:
            flag = self.cap.open(self.CAM_NUM)
            if flag is False:  # Failed to open camera
                msg = QtWidgets.QMessageBox.warning(self, u"Warning",
                                                    u"Please check your camera connection！ ",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)

            else:
                self.textEdit_pic.setText('No photos were selected')
                QtWidgets.QApplication.processEvents()
                self.textEdit_camera.setText('Camera on...')
                self.label_face.setText('Analysing...\n\nleading')
                # use recognize model
                self.emotion_model = Emotion_Rec(self.model_path)
                QtWidgets.QApplication.processEvents()
                # timer start
                self.timer_camera.start(30)
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.textEdit_camera.setText('Camera off...')
            self.textEdit_pic.setText('No photos were selected')
            self.label_face.clear()
            gif = QMovie(':/images/ui/scan.gif')
            self.label_face.setMovie(gif)
            gif.start()
            self.showResult()

    def show_camera(self):
        flag, self.image = self.cap.read()  # get Real-time image
        self.image = cv2.flip(self.image, 1)

        time_start = time.time()
        # predict
        results, result = self.emotion_model.run(self.image, self.label_face)
        time_end = time.time()
        # show result
        self.showResult(results, result, time_end - time_start)

    def choose_pic(self):
        # choose photo to test
        self.timer_camera.stop()
        self.cap.release()
        self.label_face.clear()
        self.textBrowser_result.setText('None')
        self.textBrowser_time.setText('0 s')
        self.textEdit_camera.setText('Camera off')
        self.showResult()

        # choose a photo
        fileName_choose, filetype = QFileDialog.getOpenFileName(
            self, "Choose a photo...",
            self.path,  # start path
            "Image(*.jpg;*.jpeg;*.png)")  # type
        self.path = fileName_choose
        if fileName_choose != '':
            self.textEdit_pic.setText(fileName_choose)
            self.label_face.setText('Analysing...\n\nleading')
            QtWidgets.QApplication.processEvents()
            # use model to test
            self.emotion_model = Emotion_Rec(self.model_path)

            image = self.cv_imread(fileName_choose)
            QtWidgets.QApplication.processEvents()
            time_start = time.time()
            results, result = self.emotion_model.run(image, self.label_face)
            time_end = time.time()
            self.showResult(results, result, time_end - time_start)

        else:
            # return to init status
            self.textEdit_pic.setText('No photos were selected')
            gif = QMovie(':/images/ui/scan.gif')
            self.label_face.setMovie(gif)
            gif.start()
            self.showResult()

    def cv_imread(self, filePath):
        # read image from path
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
        return cv_img

    def showResult(self, results=None, result='none', totalTime=0.0):
        if results is None:
            for emotion in self.EMOTIONS:
                bar_widget = self.findChild(QProgressBar, name='progressBar_' + emotion)
                bar_widget.setValue(0)
                text_widget = self.findChild(QTextBrowser, name='textBrowser_' + emotion)
                text_widget.setText(str(0) + '%')
        else:
            self.textBrowser_result.setText(result)
            self.textBrowser_time.setText(str(round(totalTime, 3)) + ' s')
            for (i, (emotion, prob)) in enumerate(results):
                bar_widget = self.findChild(QProgressBar, name='progressBar_' + emotion)
                bar_widget.setValue(prob * 100)
                text_widget = self.findChild(QTextBrowser, name='textBrowser_' + emotion)
                prob = round(prob * 100, 2)
                text_widget.setText(str(prob) + '%')