import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle

from train_util import train_model
from issm import load_train_dataset, load_test_dataset
from network import NormalConvolutionModelRelu
from show_history import plot_history

if os.path.isdir("../model") == False:
  os.mkdir("../model")

prepared = True
epochs = 50
device_name = "cuda"
learning_rate = 0.001
batch_size = 8
image_size = (200, 200)
model = NormalConvolutionModelRelu(image_size=image_size)

device = torch.device(device_name)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

x_train, y_train, x_validation, y_validation = load_train_dataset()

print("Start Training")
model, history = train_model(model, criterion, optimizer, epochs, x_train, y_train, x_validation, y_validation, device, batch_size)

with open("../model/model_history.pkl", "wb") as f:
  pickle.dump(history, f)

plot_history(history)

torch.save({"model_state_dict": model.state_dict()}, "../model/model_final.pth")
