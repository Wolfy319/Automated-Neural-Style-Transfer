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

# set up gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tensors to normalize inputs for vgg model
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

print(cnn_normalization_mean)
wat = torch.tensor(cnn_normalization_mean).view(-1,1,1)
print(wat)

# initialize vgg model
cnn = models.vgg19(pretrained=True).features.to(device).eval()
print(cnn)
for child in cnn :
  print(child)

imsize = 512 if torch.cuda.is_available() else 128 

# scales input and converts to tensor
loader = transforms.Compose([
    transforms.Resize(imsize),  
    transforms.ToTensor()])  

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("./data/images/neural-style/picasso.jpg")
content_img = image_loader("./data/images/neural-style/dancing.jpg")


