import torch
import torch.nn.functional as F
import pyprind
import os
import sys
from torch.utils.tensorboard import SummaryWriter
import shutil

def train_model(model, criterion, optimizer, epochs, x_train, y_train, x_test, y_test, device, batch_size):
  shutil.rmtree("../logs", ignore_errors=True)
  os.mkdir("../logs")
  writer = SummaryWriter("../logs")

  history = {"loss": [], "accuracy": []}

  for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    bar = pyprind.ProgBar(y_train.shape[0] // batch_size, track_time=True, title="Training Model")

    for i in range(y_train.shape[0] // batch_size):
      # get the inputs; data is a list of [inputs, labels]
      inputs = torch.tensor(x_train[i * batch_size : (i + 1) * batch_size], device=device, dtype=torch.float32)
      labels = torch.tensor(y_train[i * batch_size : (i + 1) * batch_size], device=device, dtype=torch.long)

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      bar.update()

    torch.save({"model_state_dict": model.state_dict()}, "../model/model_{}.pth".format(epoch))


    with torch.no_grad():
      x_test = torch.tensor(x_test[:(y_test.shape[0] // batch_size * batch_size)], device=device, dtype=torch.float32)
      y_test = torch.tensor(y_test[:(y_test.shape[0] // batch_size * batch_size)], device=device, dtype=torch.float32)

      correct = 0

      for i in range(y_test.shape[0] // batch_size):
        inputs = torch.tensor(x_test[i * batch_size : (i + 1) * batch_size], device=device, dtype=torch.float32)
        labels = torch.tensor(y_test[i * batch_size : (i + 1) * batch_size], device=device, dtype=torch.long)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

      accuracy = correct / y_test.shape[0]
      loss = running_loss / (y_train.shape[0] // batch_size)

      history["loss"].append(loss)
      history["accuracy"].append(accuracy)

      writer.add_scalar("Train Loss", loss, epoch)
      writer.add_scalar("Test Accuracy", accuracy * 100, epoch)

      print("Epoch: {} Loss: {:.3f}".format(epoch + 1, loss))
      print("Accuracy of the network on the 600 test images: %d %%" % (accuracy * 100))

  print("Finished Training")
  return model, history