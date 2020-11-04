import torch
from network import NormalConvolutionModelRelu
from issm import load_test_dataset
import numpy as np
import pyprind
import pandas as pd
import os
import torch

dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/ISSM/"

x_test, test_ids = load_test_dataset()
x_test, test_ids = x_test, test_ids
x_test = torch.tensor(x_test, dtype=torch.float32)

model = NormalConvolutionModelRelu((200, 200))
model.load_state_dict(torch.load("../model/model_10.pth"), strict=False)

softmax = torch.nn.Softmax(dim=1)

submission = np.empty((test_ids.shape[0]))
bar = pyprind.ProgBar(test_ids.shape[0], track_time=True, title="Predicting Test Images")

for image, test_id in zip(x_test, test_ids):
  image = torch.reshape(image, (1, 3, 200, 200))
  predict = softmax(model.forward(image))
  submission[test_id - 1] = int(np.argmax(predict.detach().numpy()) + 1)
  bar.update()

submission_csv = pd.read_csv(os.path.join(dataset_path, "sampleSolution.csv"))
submission_csv["LABEL"] = np.array(submission, dtype=np.int32)
submission_csv.to_csv("../submission/submission_10.csv", index=False)