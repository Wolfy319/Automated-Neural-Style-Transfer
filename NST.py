###
# Copyright (c) 2022 Wolfy Fiorini
# All rights reserved
###

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class Normalization(nn.Module) :
  def __init__(self, mean, std) :
    super(Normalization, self).__init__()
    self.mean = torch.tensor(mean).view(-1,1,1)
    self.std = torch.tensor(std).view(-1,1,1)

def forward(self, img) :
    print("Before")
    print(img)
    print("After")
    print((img - self.mean) / self.std)

    return (img - self.mean) / self.std

	
class ContentLoss(nn.Module) :
  def __init__(self, target) :
    super(ContentLoss, self).__init__()
    self.target = target.detach()
  
  def forward(self, input) :
    self.loss = F.mse_loss(input, self.target)
    return input


def imageLoader(imageName):
  image = Image.open(imageName)
  image = loader(image).unsqueeze(0)
  return image.to(device, torch.float)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vggNormalizationMean = torch.tensor([0.485, 0.456, 0.406]).to(device)
vggNormalizationStd = torch.tensor([0.229, 0.224, 0.225]).to(device)

cnn = models.vgg19(pretrained=True).features.to(device).eval()

contentLayers = ["conv_4_2"]
styleLayers = ["conv_1_1", "conv_2_1", "conv_3_1", "conv_4_1", "conv_5_1"]

imWidth = 800 if torch.cuda.is_available() else 180  # use small size if no gpu
imHeight = 600 if torch.cuda.is_available() else 135

loader = transforms.Compose([
    transforms.Resize((imHeight, imWidth)),  
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
unloader = transforms.ToPILImage()

styleImg = imageLoader("/content/crazy.jpg")
contentImg = imageLoader("/content/mountains.jpg")
inputImg = contentImg.clone()


