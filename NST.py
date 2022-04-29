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

# calculates the loss between an input image and a style image
class StyleLoss(nn.Module) :
  def __init__(self, target) :
    super(StyleLoss, self).__init__()
    self.target = GramMatrix(target).detach()

  def forward(self, input) :
    self.loss = F.l1_loss(GramMatrix(input), self.target)
    return input

# calculates the loss between an input image and a content image
class ContentLoss(nn.Module) :
  def __init__(self, target) :
    super(ContentLoss, self).__init__()
    self.target = target.detach()
  
  def forward(self, input) :
    self.loss = F.mse_loss(input, self.target)
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

def GramMatrix(input) :
  a,b,c,d = input.size()
  input = input.view(a*b, c*d)
  gram = torch.mm(input, input.t())
  return gram.div(a*b*c*d)

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer

def buildModel(cnn, mean, std, contentImage, styleImage, contentLayers, styleLayers) :
  cnn = cnn
  # initialize empty model and add normalization layer
  normalize = Normalization(mean, std)
  model = nn.Sequential(normalize)
  model.add_module("normal", normalize)

  contentLosses = []
  styleLosses = []
  i = 1
  j = 0
  # iterate through vgg layers and add to model
  for layer in cnn.children() :
    if isinstance(layer, nn.Conv2d) :
      j += 1
      name = "conv_{}_{}".format(i,j)
    # replace inplace ReLU with not-inplace ReLU
    elif isinstance(layer, nn.ReLU) :
      name = "relu_{}".format(i)
      layer = nn.ReLU(inplace=False)
    # replace MaxPool2d with AvgPool2d for better image results
    elif isinstance(layer, nn.MaxPool2d) :
      name = "avgpool_{}".format(i)
      layer = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
      i += 1
      j = 0
    model.add_module(name, layer)

    # add content loss module after a content layer
    if name in contentLayers :
      name = "content_loss_{}".format(i)
      target = model(contentImage).detach()
      contentLoss = ContentLoss(target)
      model.add_module(name, contentLoss)
      contentLosses.append(contentLoss)
    # add style loss module after a style layer
    elif name in styleLayers :
      name = "style_loss_{}".format(i)
      target = model(styleImage).detach()
      styleLoss = StyleLoss(target)
      styleLosses.append(styleLoss)
      model.add_module(name, styleLoss)

    # find the last loss module
    for i in range(len(model) - 1, -1, -1) :
      if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss) :
        break
    
    # remove any layers after the last loss layer
    model = model[:(i+1)]

    return model, contentLosses, styleLosses

  
def runModel(inputImg, contentImg, styleImg, contentLayers, styleLayers, numSteps = 300, styleWeight=1000000, contentWeight=1) :
  # build nst model
  model, cLosses, sLosses = buildModel(cnn, cnn_normalization_mean, cnn_normalization_std, contentImg, styleImg, contentLayers, styleLayers)
  model.requires_grad_(False)
  inputImg.requires_grad_(True)

  # initialize optimizer
  optimizer = get_input_optimizer(inputImg)

  # "training" loop
  step = [0]
  while step[0] <= numSteps :
    def closure() :
      with torch.no_grad() :
        inputImg.clamp_(0,1)

      optimizer.zero_grad()
      # put input through model
      model(inputImg)

      totalContentLoss = 0
      totalStyleLoss = 0

      # sum the total content and style loss, and factor in weights
      for cLoss in cLosses :
        totalContentLoss += cLoss.loss
      for sLoss in sLosses :
        totalStyleLoss += sLoss.loss

      totalContentLoss *= contentWeight
      totalStyleLoss *= styleWeight

      totalLoss = totalContentLoss + totalStyleLoss
      totalLoss.backward()

      # output progress
      step += 1
      if step % 50 == 0 :
        print("Step {}".format(step))
        print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                      totalStyleLoss.item(), totalContentLoss.item()))
        print()
      return totalContentLoss + totalStyleLoss

    # optimize
    optimizer.step(closure())

  with torch.no_grad() :
    inputImg.clamp_(0,1)
  
  return inputImg

# set up gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tensors to normalize inputs for vgg model
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

# initialize vgg model
cnn = models.vgg19(pretrained=True).features.to(device).eval()

contentLayers = ["conv_4_2"]
styleLayers = ["conv_1_1", "conv_2_1", "conv_3_1", "conv_4_1", "conv_5_1"]


# load images
style_img = image_loader("/content/points.jpg")
content_img = image_loader("/content/bird.jpg")
inputImg = content_img.clone()
output = runModel(inputImg, content_img, style_img, contentLayers, styleLayers, numSteps=300)

plt.ion()

plt.figure()
imshow(style_img, "style")
plt.pause(0.01)

plt.figure()
imshow(content_img, "content")
plt.pause(0.01)