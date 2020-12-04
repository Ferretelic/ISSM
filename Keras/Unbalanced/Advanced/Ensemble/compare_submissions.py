import os
import pandas as pd
import numpy as np
from issm import load_test_dataset
from keras.utils import to_categorical
from keras.models import load_model
import tensorflow as tf
import numpy as np
import pandas as pd
import json

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def ensemble_models(models, id):
  _, test_ids = load_test_dataset(size=64)
  predictions = np.zeros((test_ids.shape[0], 10))
  submission = np.empty((test_ids.shape[0]))

  count = 0
  for model_data in models:
    x_test, _ = load_test_dataset(size=int(model_data[0][-2:]))
    print("Predict {}".format(model_data[0]))
    count += 1
    model = load_model("../{}/models/model_{}.h5".format(model_data[0], model_data[1]))
    prediction = model.predict(x_test)
    predictions += prediction

  predictions = np.argmax(predictions / count, axis=1)

  for label, image_id in zip(predictions, test_ids):
    submission[image_id - 1] = label + 1

  submission_csv = pd.read_csv("/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/ISSM/sampleSolution.csv")
  submission_csv["LABEL"] = np.array(submission, dtype=np.int32)
  submission_csv.to_csv("./submission/submission_{}.csv".format(id), index=False)

ensemble_id = "5"
with open("./submission/submissions.json", "r") as f:
  models = json.load(f)[ensemble_id]

ensemble_models(models, ensemble_id)