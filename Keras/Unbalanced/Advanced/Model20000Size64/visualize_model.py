import keras
from model import cnn_model

model = cnn_model((64, 64, 3))
keras.utils.plot_model(model, to_file="./model.png")
