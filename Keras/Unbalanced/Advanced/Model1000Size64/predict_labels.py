from issm import load_test_dataset
from keras.utils import to_categorical
from keras.models import load_model
import tensorflow as tf
import numpy as np
import pandas as pd

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def create_submission_file(epoch):
  x_test, test_ids = load_test_dataset()

  model = load_model("./models/model_{}.h5".format(epoch))
  predictions = np.argmax(model.predict(x_test), axis=1)

  submission = np.empty((predictions.shape[0]))

  for label, image_id in zip(predictions, test_ids):
    submission[image_id - 1] = label + 1

  submission_csv = pd.read_csv("/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/ISSM/sampleSolution.csv")
  submission_csv["LABEL"] = np.array(submission, dtype=np.int32)
  submission_csv.to_csv("./submission/submission_{}.csv".format(epoch), index=False)

for epoch in ["100", "final"]:
  create_submission_file(epoch)