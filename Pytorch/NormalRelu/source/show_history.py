import matplotlib.pyplot as plt
import numpy as np

def plot_history(history):
  accuracy = history["accuracy"]
  loss = history["loss"]

  plt.figure()
  plt.plot(np.arange(len(accuracy)), accuracy)
  plt.savefig("../image/accuracy.png")

  plt.figure()
  plt.plot(np.arange(len(loss)), loss)
  plt.savefig("../image/loss.png")
