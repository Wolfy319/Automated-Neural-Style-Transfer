###
# Copyright (c) 2022 Wolfy Fiorini
# All rights reserved
###

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])


print(cnn_normalization_mean)
wat = torch.tensor(cnn_normalization_mean).view(-1,1,1)
print(wat)