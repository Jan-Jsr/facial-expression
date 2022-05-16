import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

dataset_path = 'dataset/fer2013/fer2013.csv'
image_size = (48, 48)


def visualizeImage(X_train, y_train):
    plt.rcParams['figure.figsize'] = (7.0, 5.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    classes = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
    num_classes = len(classes)
    samples_per_class = 5

    for y, cls in enumerate(classes):
        # get index of training label
        idxs = np.flatnonzero(y_train == y)
        # random choice to plot
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        # img show
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            img = X_train[idx]
            img = np.asarray(img).reshape(48, 48)
            plt.imshow(img)
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()


def load_fer2013():
    # load only 6000 data to Shorten training time
    data = pd.read_csv(dataset_path, nrows=6000)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []

    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), image_size)
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)

    # visualize for test
    #emotions = np.array(data['emotion'])
    #visualizeImage(faces, emotions)

    # get_dummies
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    return faces, emotions


# preprocess and normalize
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
