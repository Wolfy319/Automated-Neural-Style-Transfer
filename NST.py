###
# Copyright (c) 2022 Wolfy Fiorini
# All rights reserved
##

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models

# calculates the loss between an input image and a content image
class ContentLoss(nn.Module) :
  def __init__(self, target) :
    super(ContentLoss, self).__init__()
    self.target = target.detach()
  
  def forward(self, input) :
    self.loss = F.l1_loss(input, self.target)
    return input

# normalizes an input to fit the vgg19 model
class Normalization(nn.Module) :
  def __init__(self, mean, std) :
    super(Normalization, self).__init__()
    self.mean = torch.tensor(mean).view(-1,1,1)
    self.std = torch.tensor(std).view(-1,1,1)
  
  def forward(self, img) :
    return (img - self.mean) / self.std

##################################################

imWidth = 800 if torch.cuda.is_available() else 180 
imHeight = 600 if torch.cuda.is_available() else 135

loader = transforms.Compose([
    transforms.Resize((imHeight, imWidth)),  
    transforms.ToTensor()]) 
unloader = transforms.ToPILImage()

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(input, title) :
  image = input.cpu().clone()
  image = image.squeeze(0)
  image = unloader(image)
  plt.imshow(image)
  plt.title(title)

# set up gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tensors to normalize inputs for vgg model
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

# initialize vgg model
cnn = models.vgg19(pretrained=True).features.to(device).eval()



style_img = image_loader("/content/crazy.jpg")
content_img = image_loader("/content/mountains.jpg")

plt.ion()

plt.figure()
imshow(style_img, "style")
plt.pause(0.01)

plt.figure()
imshow(content_img, "content")
plt.pause(0.01)