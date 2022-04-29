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

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(input, title) :
  image = input.cpu().clone()
  image = image.squeeze(0)
  image = transforms.ToPILImage(image)
  plt.imshow(image)
  plt.title(title)
  
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

imWidth = 800 if torch.cuda.is_available() else 180  
imHeight = 600 if torch.cuda.is_available() else 135

# scales input and converts to tensor
loader = transforms.Compose([
    transforms.Resize((imHeight, imWidth)),  
    transforms.ToTensor()])  




style_img = image_loader("/content/crazy.jpg")
content_img = image_loader("/content/mountains.jpg")



plt.figure()
imshow(style_img, "style")
plt.pause(0.01)

plt.figure()
imshow(content_img, "content")
plt.pause(0.01)