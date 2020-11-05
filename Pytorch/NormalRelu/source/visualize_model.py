import torch
from distribution_3d import plot_distribution
from network import DistributionConvolutionModelKReluGradientMuRandom
import matplotlib.pyplot as plt

model = DistributionConvolutionModelKReluGradientMuRandom()
model.load_state_dict(torch.load("../model/model_final.pth"), strict=False)
distribution = model.conv3_1.distribution.detach().numpy()[2]

plot_distribution(distribution)