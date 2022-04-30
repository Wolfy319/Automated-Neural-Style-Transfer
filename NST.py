###
# Copyright (c) 2022 Wolfy Fiorini
# All rights reserved
##

NST attempt #3.ipynb

Code cell <Aef5qSRevgnA>
#%% [code]
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
device = "cpu"
print(device)
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

def imshow(input, title) :
  image = input.cpu().clone()
  image = image.squeeze(0)
  image = unloader(image)
  plt.imshow(image)
  plt.title(title)



content_image = imageLoader("mountains.jpg")
# style_image = imageLoader("crazy.jpg")
# input_image = content_image.clone().to(device)



# plt.ion()
# plt.figure()
# imshow(content_image, "content")
# plt.pause(0.01)

# plt.figure()
# imshow(style_image, "style")
# plt.pause(0.01)
