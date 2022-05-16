import warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from load_and_process import load_fer2013
from load_and_process import preprocess_input
from sklearn.model_selection import train_test_split
from models.cnn import mini_XCEPTION
import time


batch_size = 32
num_epochs = 1000
input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50
base_path = 'models/emotion_models/'
time1 = time.process_time()

# load dataset
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
time2 = time.process_time()

# 20% for test
xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.2, shuffle=True)
time3 = time.process_time()

# use our model for train
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam',  # adam optimizer
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
time4 = time.process_time()


log_file_path = base_path + '_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience / 4),
                              verbose=1)

# path for model to save
trained_models_path = base_path + '_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'

model_checkpoint = ModelCheckpoint(model_names,
                                   'val_loss', verbose=1,
                                   save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# data generator
data_generator = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True)

# train
time5 = time.process_time()
model.fit_generator(data_generator.flow(xtrain, ytrain, batch_size),
                    steps_per_epoch=len(xtrain) / batch_size,
                    epochs=num_epochs,
                    verbose=1, callbacks=callbacks,
                    validation_data=(xtest, ytest))
time6 = time.process_time()

print('load data time：', round(time2 - time1, 4))
print('modeling time：', round(time4 - time3, 4))
print('training time：', round(time6 - time5, 4))
