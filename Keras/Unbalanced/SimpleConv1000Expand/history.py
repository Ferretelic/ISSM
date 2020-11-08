import numpy as np
import matplotlib.pyplot as plt

def show_history(history):
  epochs = np.arange(len(history["loss"]))

  train_loss = history["loss"]
  validation_loss = history["val_loss"]

  plt.figure()
  plt.plot(epochs, train_loss, label="train loss")
  plt.plot(epochs, validation_loss, label="validation loss")
  plt.savefig("./history/loss.png")

  train_accuracy = history["accuracy"]
  validation_accuracy = history["val_accuracy"]

  plt.figure()
  plt.plot(epochs, train_accuracy, label="train accuracy")
  plt.plot(epochs, validation_accuracy, label="validation accuracy")
  plt.savefig("./history/accuracy.png")

import pickle

with open("./history/model_history.pkl", "rb") as f:
  history = pickle.load(f)
