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


device = torch.device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device = "cpu"
print(device)
vggNormalizationMean = torch.tensor([0.485, 0.456, 0.406]).to(device)
vggNormalizationStd = torch.tensor([0.229, 0.224, 0.225]).to(device)

imgWidth = 800 if device == "cuda" else 180
imgHeight = 600 if device == "cuda" else 135

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

def imshow(input, title, mean, std) :
  mean = torch.tensor(mean).view(-1,1,1)
  std = torch.tensor(std).view(-1,1,1)
  image = input.cpu().clone()
  image = image.squeeze(0)
  image = unloader(image)
  plt.imshow(image)
  plt.title(title)

content_image = imageLoader("mountains.jpg")
style_image = imageLoader("crazy.jpg")
input_image = content_image.clone().to(device)



plt.ion()
plt.figure()
imshow(content_image, "content")
plt.pause(0.01)

plt.figure()
imshow(style_image, "style")
plt.pause(0.01)


class NST(nn.Module) :
  def __init__(self) :
    super(NST, self).__init__()
    self.style_features = [] 
    self.content_features = []
    self.style_layers = [0, 5, 10, 19, 28]
    self.content_layers = [21]
    self.model = models.vgg19(pretrained=True).features[:29].to(device).eval()

  def forward(self, x) :
    self.style_features = []
    self.content_features = []
    for layer_num, layer in enumerate(self.model) :
      
      out = layer(x)
      x = out
      if layer_num in self.style_layers :
        self.style_features.append(x)
      elif layer_num in self.content_layers :
        self.content_features.append(x)
    return x

def build_model(content, style) :
  model = NST().to(device).eval()
  for layer_num, layer in enumerate(model.model) :
    if isinstance(layer, nn.MaxPool2d) :
      layer = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
      model.model[layer_num] = layer
    elif isinstance(layer, nn.ReLU) :
      layer = nn.ReLU(inplace=False)
      model.model[layer_num] = layer
  style = Normalize(style, vggNormalizationMean,vggNormalizationStd)
  content = Normalize(style, vggNormalizationMean,vggNormalizationStd)
  model(style)
  style_features = model.style_features
  model(content)
  content_features = model.content_features 
  return model, style_features, content_features

def transfer(image, content, style, loss_fn) :
  vggNormalizationMean = torch.tensor([0.485, 0.456, 0.406]).to(device)
  vggNormalizationStd = torch.tensor([0.229, 0.224, 0.225]).to(device)
  # Initialize model
  model, style_features, content_features = build_model(content, style)
  model.requires_grad_(False)

  image.requires_grad_(True)
  # Initialize opt
  optimizer = optim.Adam([image], lr=lr)

  ## training loop
  for step in range(num_steps) :
    # image = Normalize(image, vggNormalizationMean, vggNormalizationStd)
    with torch.no_grad() :
        image.clamp_(0,1)
    model(image)
    input_style = model.style_features
    input_content = model.content_features
    style_loss = 0
    content_loss = 0
    # calc style and content loss
    i = 0
    for feature, map in zip(style_features, input_style) :
      target = GramMatrix(feature)
      generated = GramMatrix(map)
      style_loss += F.mse_loss(generated, target)
      i += 1
    for feature, map in zip(content_features, input_content) :
      content_loss += F.mse_loss(map, feature)
    style_loss *= style_weight
    content_loss *= content_weight

    total_loss = content_loss + style_loss
    optimizer.zero_grad()
    total_loss.backward(retain_graph=True)
    optimizer.step()

    print("Step {}/{} - Loss : {}".format(step, num_steps, total_loss))

  with torch.no_grad() :
        image.clamp_(0,1)
  return image


def GramMatrix(input) :
  _,channels,height,width = input.size()
  features = input.view(channels, height*width)
  gram = torch.mm(features, features.t())
  return gram.div(channels*height*width)


def Normalize(input, mean, std) :
  mean = torch.tensor(mean).view(-1,1,1)
  std = torch.tensor(std).view(-1,1,1)

  return (input - mean) / std

# hyperparams
loss_fn = F.l1_loss
num_steps = 50
lr = 0.5
style_weight = 0.1
content_weight = 1000
result = transfer(input_image, content_image, style_image, loss_fn)
plt.figure()
imshow(result, "result")
plt.pause(0.01)


