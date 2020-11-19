from issm import load_regulized_train_dataset, load_test_dataset
from keras.utils import to_categorical
from model import cnn_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 64
epochs = 200

x_train, y_train = load_regulized_train_dataset()
y_train = to_categorical(y_train)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2)

train_generator = ImageDataGenerator(rotation_range = 20, horizontal_flip = True, height_shift_range = 0.2, width_shift_range = 0.2,zoom_range = 0.2, channel_shift_range = 0.2)
validation_generator = ImageDataGenerator()

model = cnn_model()

callbacks = [
  ModelCheckpoint(filepath="./models/model_{epoch:02d}.h5"),
  TensorBoard(log_dir="./logs"),
]

model_history = model.fit_generator(train_generator.flow(x_train, y_train, batch_size), epochs=epochs, callbacks=callbacks, validation_data=validation_generator.flow(x_validation, y_validation, batch_size))

model.save("./models/model_final.h5")
history = model_history.history

with open("./model_history.pkl", "wb") as f:
  pickle.dump(history, f)
