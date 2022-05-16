import warnings
import os
from time import process_time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.models import load_model
from load_and_process import load_fer2013, preprocess_input
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')

# use our test set to test the trained model and show the confusion matrix

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
emotion_model_path = 'models/emotion_models/_mini_XCEPTION.102-0.66.hdf5'
emotion_model = load_model(emotion_model_path, compile=False)  # load model
input_shape = emotion_model.input_shape[1:]  # input shape

time1 = process_time()

faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
time2 = process_time()

# 20% for the test set
xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.2, shuffle=True)

ndata = xtest.shape[0]
y_pred = np.zeros((ndata,))
y_true = [ytest[i].argmax() for i in range(ndata)]
y_true = np.array(y_true)

# test the model
for i in range(ndata):
    input_image = xtest[i]
    input_image = cv2.resize(input_image, input_shape[0:2], cv2.INTER_NEAREST)
    input_image = np.reshape(input_image, (1, input_shape[0], input_shape[1], input_shape[2]))
    preds = emotion_model.predict(input_image)[0]
    y_pred[i] = preds.argmax()

tick_marks = np.array(range(len(EMOTIONS))) + 0.5


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(EMOTIONS)))
    plt.xticks(xlocations, EMOTIONS, rotation=45)
    plt.yticks(xlocations, EMOTIONS)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # accuracy for each label
print('Confusion matirx：')
print(cm_normalized)
accuracy = np.mean([cm_normalized[i, i] for i in range(num_classes)])
print('Accuracy：' + str(round(accuracy, 2)))

# plot
plt.figure(figsize=(12, 8), dpi=120)
ind_array = np.arange(len(EMOTIONS))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(False, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
plt.savefig('temp/confusion_matrix.png', format='png')
plt.show()

print('time：', round(time2 - time1, 4))
