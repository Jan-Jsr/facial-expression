"""
Libraries that need to be installed：
    keras
    PyQt5
    pandas
    scikit-learn
    tensorflow
    imutils
    opencv-python
    matplotlib

run main.py
"""

import warnings
import os

from ui_mainWindow import MainWindow
from sys import argv, exit
from PyQt5.QtWidgets import QApplication

# ignore warining
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    app = QApplication(argv)
    window = MainWindow()
    window.show()
    exit(app.exec_())
