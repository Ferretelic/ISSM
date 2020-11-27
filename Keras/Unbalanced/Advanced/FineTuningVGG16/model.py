from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.applications import VGG16

def cnn_model(input_size):
  model = Sequential()

  vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=input_size)
  for layer in vgg16.layers:
    layer.trainable = False
  model.add(vgg16)
  model.add(GlobalAveragePooling2D())
  model.add(Dense(1024, activation="relu"))
  model.add(Dropout(0.5))
  model.add(Dense(1024, activation="relu"))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation="softmax"))

  model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

  model.summary()
  return model
