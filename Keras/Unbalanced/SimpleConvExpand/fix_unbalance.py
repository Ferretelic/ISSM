from keras.preprocessing.image import ImageDataGenerator
from issm import load_train_dataset
import numpy as np
import pickle
import os

def under_sampling(images, labels, sample):
  index = np.arange(images.shape[0])
  np.random.shuffle(index)
  sampled_data = images[index[:sample]]

  labels = np.ones((sample,)) * labels[0]

  return sampled_data, labels

def over_sampling(images, labels, sample):
  generator = ImageDataGenerator(rotation_range = 20, horizontal_flip = True, height_shift_range = 0.2, width_shift_range = 0.2,zoom_range = 0.2, channel_shift_range = 0.2)

  image_counts = images.shape[0]
  ratio = (sample // image_counts) + 1
  generator = generator.flow(images, labels, batch_size=labels.shape[0])
  generated_images = np.empty((ratio * image_counts,) + images.shape[1:])

  for i in range(ratio):
    generated_data = next(generator)
    generated_images[(i * image_counts):((i + 1) * image_counts)] = generated_data[0]

  generated_images = np.array(generated_images[:sample])
  labels = np.ones((sample,)) * labels[0]

  return generated_images, labels

def regulation_dataset(classes=10, sample=500):
  x_train, y_train = load_train_dataset()

  all_image_counts = sample * classes
  all_images = np.empty((all_image_counts,) + x_train.shape[1:])
  all_labels = np.empty((all_image_counts,))

  for label in range(classes):
    index = np.where(y_train == label, True, False)
    images = x_train[index]
    labels = y_train[index]

    if labels.shape[0] > sample:
      images, labels = under_sampling(images, labels, sample)
    else:
      images, labels = over_sampling(images, labels, sample)

    all_images[label * sample: (label + 1) * sample] = images
    all_labels[label * sample: (label + 1) * sample] = labels

  all_images = np.array(all_images, dtype=np.float32)
  all_labels = np.array(all_labels, dtype=np.int32)

  dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/ISSM/"

  with open(os.path.join(dataset_path, "regulized_train_images.pkl"), "wb") as f:
    pickle.dump(all_images, f)

  with open(os.path.join(dataset_path, "regulized_train_labels.pkl"), "wb") as f:
    pickle.dump(all_labels, f)