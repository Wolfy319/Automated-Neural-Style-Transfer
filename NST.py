###
# Copyright (c) 2022 Wolfy Fiorini
# All rights reserved
##

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
import numpy as np


device = "cpu"
vggNormalizationMean = torch.tensor([0.485, 0.456, 0.406]).to(device)
vggNormalizationStd = torch.tensor([0.229, 0.224, 0.225]).to(device)



vggNormalizationMean = torch.tensor([0.485, 0.456, 0.406]).to(device)
vggNormalizationStd = torch.tensor([0.229, 0.224, 0.225]).to(device)

imgWidth = 500 if device == "cuda" else 250
imgHeight = 500 if device == "cuda" else 250

loader = transforms.Compose([
    transforms.Resize((imgHeight, imgWidth)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), 
                          (0.229, 0.224, 0.225))
    ]
)
unloader = transforms.ToPILImage()

def imageLoader(imageName):
  image = Image.open(imageName)
  image = loader(image).unsqueeze(0)
  return image.to(device)

def imshow(input, title, mean, std) :
  mean = mean.view(-1,1,1)
  std = std.view(-1,1,1)
  image = input.cpu().clone()
  image = image.squeeze(0)
  image = (image * mean) + std
  image = image.clip(0, 1)
  image = unloader(image)

  plt.imshow(image)
  plt.title(title)



content_image = imageLoader("skyline.jpg")
style_image = imageLoader("art.jpeg")



plt.ion()
plt.figure()
imshow(content_image, "content", vggNormalizationMean, vggNormalizationStd)
plt.pause(0.01)

imshow(style_image, "style", vggNormalizationMean, vggNormalizationStd)
plt.pause(0.01)

cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn.requires_grad_(False)

layers = {"style" : {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '30': 'conv5_2',
                  '28': 'conv5_1'}, 
          "content" : {'30' : 'conv4_2'}}

input_image = content_image.clone().to(device)

def get_feature_maps(image, model, layers) :
  feature_maps = {}
  x = image
  for name, layer in model._modules.items() :
    x = layer(x)
    if name == "31" :
      break
    elif name in layers :
      feature_maps[name] = x
  return feature_maps

def style_loss(target_gram, input_features) :
  loss = 0
  for layer in layers["style"] :
    input_gram = gram(input_features[layer])
    target = target_gram[layer]
    loss += abs(torch.mean(target - input_gram)) 
  return loss

def content_loss(target_features, input_features) :
  loss = 0
  for layer in layers["content"] :
    loss += abs(torch.mean(input_features[layer] - target_features[layer]))
  return loss

def gram(input) :
  _,channels,height,width = input.size()
  features = input.view(channels, height*width)
  gram = torch.mm(features, features.t())
  return gram.div(channels * height * width)

def run_model(cnn, style, content, input, layers, steps, style_weight, content_weight, learn_rate) :
  input.requires_grad_(True)
  # Initialize NST model and optimizer
  model = cnn.requires_grad_(False)
  optimizer = optim.Adam([input], lr = learn_rate)

  # Calculate feature maps for style and content
  style_features = get_feature_maps(style, model, layers["style"])
  style_grams = {layer: gram(style_features[layer]) for layer in style_features}
  content_features = get_feature_maps(content, model, layers["content"])

  # Training loop
  for step in range(steps):
    input_style = get_feature_maps(input, model, layers["style"])
    input_content = get_feature_maps(input, model, layers["content"])
    
    total_style_loss = style_weight * style_loss(style_grams, input_style)
    total_content_loss = content_weight * content_loss(content_features, input_content)
    total_loss = total_style_loss + total_content_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 20 == 0 : 
      print("Step {}/{} - Total loss = {}, Style loss = {}, Content loss = {}".format(step, steps, total_loss, total_style_loss, total_content_loss))
      plt.figure()
      imshow(input, "Loss: {}".format(total_loss), vggNormalizationMean, vggNormalizationStd)
      plt.pause(0.01)
  return input
  
lr = 0.05
style_weight = 1e9
content_weight = 1e-2
num_steps = 1000
output = run_model(cnn, style_image, content_image, input_image, layers, num_steps, style_weight, content_weight, lr)  

print(model)


plt.figure()
imshow(output, "output", vggNormalizationMean, vggNormalizationStd)
plt.pause(0.01)


