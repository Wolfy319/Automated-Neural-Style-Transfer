###
# Copyright (c) 2022 Wolfy Fiorini
# All rights reserved
##

import os
from datetime import datetime
from torchvision.transforms.functional import normalize
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import torch.optim as optim
import torch.nn.functional as F 

device = "cuda" if torch.cuda.is_available() else "cpu"
imgWidth = 800 if device == "cuda" else 640
imgHeight = 600 if device == "cuda" else 480

loader = transforms.Compose([
    transforms.Resize((imgHeight, imgWidth)),
    transforms.ToTensor()
    ]
)
unloader = transforms.ToPILImage()



def imageLoader(imageName):
	image = Image.open(imageName)
	image = loader(image).unsqueeze(0)
	return image.to(device)
 
def imfit(input) :
  image = input.to(device).clone()
  image = image.squeeze(0)
  image = image.data.clamp(0,1)
  image = unloader(image)
  return image
 
def unnormal(input) :
  return (input - vggNormalizationMean) / vggNormalizationStd

content_image = imageLoader("skyline.jpg")
style_image = imageLoader("starry.jpg")
input_image = content_image.clone().to(device)

plt.figure()
content = imfit(content_image)
style = imfit(style_image)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("content")
ax1.imshow(content)
ax2.set_title("style")
ax2.imshow(style)
plt.show()
plt.pause(0.01)




class Normalizer(nn.Module) :
  def __init__(self) :
    super().__init__()
    self.mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    self.std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
  def forward(self, x) :
    return (x - self.mean) / self.std

class StyleLoss(nn.Module) :
  def __init__(self, target) :
    super().__init__()
    self.target = target.detach()
    self.loss = None
  
  def gram(self, input) :
    b,c,h,w = input.size()
    features = input.view(b*c, h*w)
    result = torch.mm(features, features.t())
    return result.div(b*c*h*w)

  def forward(self, input) :
    target_gram = self.gram(self.target)
    input_gram = self.gram(input)
    self.loss = F.l1_loss(input_gram, target_gram)
    return input
  
class ContentLoss(nn.Module) :
  def __init__(self, target) :
    super().__init__()
    self.target = target.detach()
    self.loss = None

  def forward(self, input) :
    self.loss = F.l1_loss(self.target, input)
    return input

vgg = models.vgg19(pretrained=True).features.to(device).eval()

def get_model_and_losses(vgg, content_image, style_image, content_layers, style_layers) :
  model = nn.Sequential(Normalizer())
  content_losses, style_losses = [],[]
  c_layers_copy, s_layers_copy = content_layers, style_layers

  i = j = 1
  for k, (name, layer) in enumerate(vgg.named_children()) :
    if isinstance(layer, nn.Conv2d) :
      name = "conv{}_{}".format(i, j)
      j += 1
    elif isinstance(layer, nn.ReLU) :
      name = "relu{}_{}".format(i,j)
      layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d) :
      name = "avgpool{}".format(i)
      layer = nn.AvgPool2d(kernel_size=2, stride=2)
      j = 0
      i += 1
    model.add_module(name, layer)
    if name in c_layers_copy :
      content_loss = ContentLoss(model(content_image))
      print(content_loss)
      model.add_module("contentloss{}".format(i), content_loss)
      content_losses.append(content_loss)
      c_layers_copy.remove(name)
    elif name in s_layers_copy :
      style_loss = StyleLoss(model(style_image))
      model.add_module("styleloss{}".format(i), style_loss)
      style_losses.append(style_loss)
      s_layers_copy.remove(name)
    if len(c_layers_copy) == 0 and len(s_layers_copy) == 0 :
      break
    
  return model, content_losses, style_losses

def run_nst(vgg, content_image, style_image, input_image, content_layers, style_layers, content_weight, style_weight, steps, learn_rate) :
  # initialize model
  nst, content_losses, style_losses = get_model_and_losses(vgg, content_image, style_image, content_layers, style_layers)
  nst.requires_grad_(False)
  optimizer = optim.Adam([input_image.requires_grad_(True)], lr = learn_rate)

  # training loop 
  for step in range(steps) :
    input_image.data.clamp(0,1)
    nst(input_image)
    content_loss = style_loss = 0
    for item in content_losses :
      content_loss += content_weight * item.loss
    for item in style_losses :
      style_loss += style_weight * item.loss
    total_loss = style_loss + content_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0 : 
        print("Step {}/{} - Total loss = {}, Style loss = {}, Content loss = {}".format(step, steps, total_loss, style_loss, content_loss))
        current_image = unnormal(input_image.to(device).clone())
        current_image = imfit(current_image)
        plt.figure()
        plt.imshow(current_image)
        plt.pause(0.01)
  input_image.data.clamp(0,1)
  return input_image()

vggNormalizationMean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1,1,1)
vggNormalizationStd = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1,1,1)

vgg_default_content_layers = ['relu3_2']
vgg_default_style_layers = [
  'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

lr = 0.01
steps = 500
style_weight = 1e9
content_weight = 1

output = run_nst(vgg, content_image, style_image, input_image, vgg_default_content_layers, vgg_default_style_layers, content_weight, style_weight, steps, lr)