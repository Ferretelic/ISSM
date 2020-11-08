import numpy as np
import cv2
import os
import pickle

def prepare_dataset():
  dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/ISSM/"
  train_path = os.path.join(dataset_path, "semTrain", "semTrain")
  test_path = os.path.join(dataset_path, "semTest", "semTest")

  train_images = []
  train_labels = []
  for category in os.listdir(train_path):
    if category[0] != ".":
      label = np.int32(category[-2:])
      for image_name in os.listdir(os.path.join(train_path, category)):
        if image_name[0] != ".":
          image = cv2.imread(os.path.join(train_path, category, image_name))
          image = (cv2.resize(image, (64, 64)) / 255.).astype(np.float32)
          train_images.append(image)
          train_labels.append(label - 1)

  test_images = []
  test_ids = []
  for image_name in os.listdir(test_path):
    if image_name[0] != ".":
      image = cv2.imread(os.path.join(test_path, image_name))
      image = (cv2.resize(image, (64, 64)) / 255.).astype(np.float32)
      image_id = np.int32(image_name[:4])

      test_images.append(image)
      test_ids.append(image_id)

  with open(os.path.join(dataset_path, "train_images.pkl"), "wb") as f:
    pickle.dump(np.array(train_images), f)

  with open(os.path.join(dataset_path, "train_labels.pkl"), "wb") as f:
    pickle.dump(np.array(train_labels), f)

  with open(os.path.join(dataset_path, "test_images.pkl"), "wb") as f:
    pickle.dump(np.array(test_images), f)

  with open(os.path.join(dataset_path, "test_ids.pkl"), "wb") as f:
    pickle.dump(np.array(test_ids), f)

def load_test_dataset():
  dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/ISSM/"

  with open(os.path.join(dataset_path, "test_images.pkl"), "rb") as f:
    x_test = pickle.load(f)

  with open(os.path.join(dataset_path, "test_ids.pkl"), "rb") as f:
    test_ids = pickle.load(f)

  return x_test, test_ids


def load_train_dataset():
  dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/ISSM/"

  with open(os.path.join(dataset_path, "train_images.pkl"), "rb") as f:
    x_train = pickle.load(f)

  with open(os.path.join(dataset_path, "train_labels.pkl"), "rb") as f:
    y_train = pickle.load(f)

  train_index = np.arange(0, y_train.shape[0])
  np.random.shuffle(train_index)

  x_train = x_train[train_index]
  y_train = y_train[train_index]

  return x_train, y_train
