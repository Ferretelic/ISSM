from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D, Dense

def cnn_model():
  model = Sequential()

  model.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=(64, 64, 3)))
  model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
  model.add(Dropout(0.25))
  model.add(MaxPooling2D())

  model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
  model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
  model.add(Dropout(0.25))
  model.add(MaxPooling2D())

  model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
  model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
  model.add(GlobalAveragePooling2D())

  model.add(Dense(1024, activation="relu"))
  model.add(Dropout(0.25))
  model.add(Dense(10, activation="softmax"))

  model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

  return model