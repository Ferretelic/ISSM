from keras.applications import VGG16
import pyprind
import numpy as np
import os
import pickle
from unbalance import load_regulized_train_dataset

def prepare_extract_dataset(x_train, size, samples, batch_size=50):
  vgg = VGG16(weights="imagenet", include_top=False, input_shape=(size, size, 3))

  bar = pyprind.ProgBar(samples // batch_size, title="Extracting")
  extracted_images = np.empty((samples, 2, 2, 512))

  for index in range(samples // batch_size):
    image_batch = x_train[index * batch_size: (index + 1) * batch_size]
    extracted = vgg.predict(image_batch)
    extracted_images[index * batch_size: (index + 1) * batch_size] = extracted
    bar.update()

  extracted_images = np.array(extracted_images)

  dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/ISSM/Extracted"

  with open(os.path.join(dataset_path, "extracted_{}_{}_vgg16.pkl".format(samples, size)), "wb") as f:
    pickle.dump(extracted_images, f)

def load_extract_dataset(size, samples):
  dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/ISSM/Extracted"

  if os.path.exists(os.path.join(dataset_path, "extracted_{}_{}_vgg16.pkl".format(samples, size))) == False:
    x_train, _ = load_regulized_train_dataset(samples, size)
    prepare_extract_dataset(x_train, size, samples)

  with open(os.path.join(dataset_path, "extracted_{}_{}_vgg16.pkl".format(samples, size)), "rb") as f:
    x_train = pickle.load(f)

    return x_train
