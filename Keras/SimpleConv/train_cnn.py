from issm import load_train_dataset, load_test_dataset
from keras.utils import to_categorical
from model import cnn_model
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import pickle
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

x_train, y_train = load_train_dataset()
y_train = to_categorical(y_train)

batch_size = 64
epochs = 200

model = cnn_model()

callbacks = [
  ModelCheckpoint(filepath="./models/model_{epoch:02d}.h5"),
  TensorBoard(log_dir="./logs"),
]

model_history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, validation_split=0.2)

model.save("./models/FinalCNN.h5")
history = model_history.history

with open("./model_history.pkl", "wb") as f:
  pickle.dump(history, f)
