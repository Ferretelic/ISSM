from layer import NormalConvolutionLayerRelu
import torch.nn as nn
import torch

class NormalConvolutionModelRelu(nn.Module):
  def __init__(self, image_size):
    super(NormalConvolutionModelRelu, self).__init__()

    # Conv1
    self.conv1_1 = NormalConvolutionLayerRelu(filters=32, kernel_size=(3, 3), input_size=((3,) + image_size), padding=1)
    self.batch_normalization1_1 = nn.BatchNorm2d(self.conv1_1.output_size[0])
    self.conv1_2 = NormalConvolutionLayerRelu(filters=32, kernel_size=(3, 3), input_size=self.conv1_1.output_size, padding=1)
    self.batch_normalization1_2 = nn.BatchNorm2d(self.conv1_2.output_size[0])
    conv1_output = self.get_pool_output(self.conv1_2)

    # Conv2
    self.conv2_1 = NormalConvolutionLayerRelu(filters=64, kernel_size=(3, 3), input_size=conv1_output, padding=1)
    self.batch_normalization2_1 = nn.BatchNorm2d(self.conv2_1.output_size[0])
    self.conv2_2 = NormalConvolutionLayerRelu(filters=64, kernel_size=(3, 3), input_size=self.conv2_1.output_size, padding=1)
    self.batch_normalization2_2 = nn.BatchNorm2d(self.conv2_2.output_size[0])
    conv2_output = self.get_pool_output(self.conv2_2)
    self.batch_normalization2 = nn.BatchNorm2d(conv2_output[1])

    ## Conv3
    self.conv3_1 = NormalConvolutionLayerRelu(filters=128, kernel_size=(3, 3), input_size=conv2_output, padding=1)
    self.batch_normalization3_1 = nn.BatchNorm2d(self.conv3_1.output_size[0])
    self.conv3_2 = NormalConvolutionLayerRelu(filters=128, kernel_size=(3, 3), input_size=self.conv3_1.output_size, padding=1)
    self.batch_normalization3_2 = nn.BatchNorm2d(self.conv3_2.output_size[0])
    conv3_output = self.get_pool_output(self.conv3_2)
    self.batch_normalization3 = nn.BatchNorm2d(conv3_output[1])

    # Linear
    self.linear_input = conv3_output[0] * conv3_output[1] * conv3_output[2]

    self.linear = nn.Linear(self.linear_input, 10)
    self.pool = nn.MaxPool2d(2, 2)
    self.drop_out = nn.Dropout(p=0.3)

  def forward(self, x):
    x = self.batch_normalization1_1(self.conv1_1.forward(x))
    x = self.batch_normalization1_2(self.conv1_2.forward(x))
    x = self.drop_out(self.pool(x))

    x = self.batch_normalization2_1(self.conv2_1.forward(x))
    x = self.batch_normalization2_2(self.conv2_2.forward(x))
    x = self.drop_out(self.pool(x))

    x = self.batch_normalization3_1(self.conv3_1.forward(x))
    x = self.batch_normalization3_2(self.conv3_2.forward(x))
    x = self.drop_out(self.pool(x))

    x = x.reshape(-1, self.linear_input)
    x = self.linear(x)
    return x

  def get_pool_output(self, layer):
    return (layer.output_size[0], layer.output_size[1] // 2, layer.output_size[2] // 2)