from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D, Dense, BatchNormalization

def cnn_model(input_size):
  model = Sequential()

  model.add(Conv2D(64, (3, 3), padding="same", activation="relu", input_shape=input_size))
  model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
  model.add(MaxPooling2D())
  model.add(Dropout(0.25))

  model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
  model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
  model.add(BatchNormalization())
  model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
  model.add(MaxPooling2D())
  model.add(Dropout(0.25))

  model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
  model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
  model.add(BatchNormalization())
  model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
  model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
  model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
  model.add(BatchNormalization())
  model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
  model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
  model.add(GlobalAveragePooling2D())

  model.add(Dense(1024, activation="relu"))
  model.add(Dropout(0.5))
  model.add(Dense(1024, activation="relu"))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation="softmax"))

  model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

  return model