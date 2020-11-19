from unbalance import load_regulized_train_dataset
from history import show_history
from keras.utils import to_categorical
from model import cnn_model
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 64
epochs = 200
samples = 2000

x_train, y_train = load_regulized_train_dataset(samples)
y_train = to_categorical(y_train)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)

model = cnn_model()

callbacks = [
  ModelCheckpoint(filepath="./models/model_{epoch:02d}.h5"),
  TensorBoard(log_dir="./logs")
]

model_history = model.fit(x_train, y_train, epochs=epochs, callbacks=callbacks, validation_data=(x_validation, y_validation))

model.save("./models/model_final.h5")
history = model_history.history

with open("./history/model_history.pkl", "wb") as f:
  pickle.dump(history, f)

show_history(history)
