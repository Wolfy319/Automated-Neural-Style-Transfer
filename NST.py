###
# Copyright 2022 Wolfy Fiorini
# All rights reserved
###

import os
from torchvision.transforms.functional import normalize
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageOps
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import myconfig as config


device = "cuda" if torch.cuda.is_available() else "cpu"
global imgWidth
imgwidth = 600 if device == "cuda" else 640
global imgHeight 
imgHeight = 600 if device == "cuda" else 480

vgg_model = models.vgg19(pretrained=True).features.to(device).eval()
vggNormalizationMean = torch.tensor(
	[0.485, 0.456, 0.406]).to(device).view(-1, 1, 1)
vggNormalizationStd = torch.tensor(
	[0.229, 0.224, 0.225]).to(device).view(-1, 1, 1)

vgg_default_content_layers = ['relu4_2']
vgg_default_style_layers = [
	'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
style_layer_weights = {'relu1_1' : 1, 'relu2_1' : 1, 'relu3_1' : 1, 'relu4_1' : 1, 'relu5_1' : 1}

lr = config.lr
steps = config.steps
style_weight = config.style_weight
content_weight = config.content_weight
numupdate = config.numupdates
numimg = config.numimg



loader = transforms.Compose([
	transforms.ToTensor()
]
)
unloader = transforms.ToPILImage()


def imageLoader(imageName):
	image = Image.open(imageName)
	image = ImageOps.fit(image, (imgWidth, imgHeight))
	image = loader(image).unsqueeze(0)
	return image.to(device)


def imfit(input):
	image = input.to(device).clone()
	image = image.squeeze(0)
	image = image.data.clamp(0, 1)
	image = unloader(image)
	return image

def create_results_folder(pathname) :
	path = "/" + pathname
	dir = os.getcwd()
	results_folder = dir + path
	os.makedirs(results_folder)
	os.makedirs(results_folder + "/Video")
	return results_folder

def create_noise_img(width, height) :
	noise = 255*torch.rand(1,3,height, width).to(device)
	noise = noise.int()
	noise = noise.float()
	return noise

class Normalizer(nn.Module):
	def __init__(self):
		super().__init__()
		self.mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).to(device)
		self.std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).to(device)

	def forward(self, x):
		return (x - self.mean) / self.std


class StyleLoss(nn.Module):
	def __init__(self, target):
		super().__init__()
		self.target = target.detach()
		self.loss = None

	def gram(self, input):
		b, c, h, w = input.size()
		features = input.view(b*c, h*w)
		result = torch.mm(features, features.t())
		return result.div(b*c*h*w)

	def forward(self, input):
		target_gram = self.gram(self.target)
		input_gram = self.gram(input)
		self.loss = F.l1_loss(input_gram, target_gram)
		return input


class ContentLoss(nn.Module):
	def __init__(self, target):
		super().__init__()
		self.target = target.detach()
		self.loss = None

	def forward(self, input):
		self.loss = F.l1_loss(self.target, input)
		return input




def get_model_and_losses(vgg, content_image, style_image, content_layers, style_layers):
	model = nn.Sequential(Normalizer())
	content_losses, style_losses = [], []
	c_layers_copy, s_layers_copy = content_layers.copy(), style_layers.copy()

	i = j = 1
	for k, (name, layer) in enumerate(vgg.named_children()):
		if isinstance(layer, nn.Conv2d):
			name = "conv{}_{}".format(i, j)
			j += 1
		elif isinstance(layer, nn.ReLU):
			name = "relu{}_{}".format(i, j)
			layer = nn.ReLU(inplace=False)
		elif isinstance(layer, nn.MaxPool2d):
			name = "avgpool{}".format(i)
			layer = nn.AvgPool2d(kernel_size=2, stride=2)
			j = 0
			i += 1
		model.add_module(name, layer)
		if name in c_layers_copy:
			content_loss = ContentLoss(model(content_image))
			model.add_module("contentloss{}".format(i), content_loss)
			content_losses.append(content_loss)
			c_layers_copy.remove(name)
		elif name in s_layers_copy:
			style_loss = StyleLoss(model(style_image))
			model.add_module("styleloss{}".format(i), style_loss)
			style_losses.append(style_loss)
			s_layers_copy.remove(name)
		if len(c_layers_copy) == 0 and len(s_layers_copy) == 0:
			break

	return model, content_losses, style_losses


def run_nst(content_image, style_image, input_image, iter, content_num, pathname, interp):
	# initialize model
	nst, content_losses, style_losses = get_model_and_losses(
		vgg_model, content_image, style_image, vgg_default_content_layers, vgg_default_style_layers)
	nst.requires_grad_(False)
	optimizer = optim.Adam([input_image.requires_grad_(True)], lr)
	name = ""
	iter *= numimg
	files = []
	# training loop
	for step in range(steps):
		input_image.data.clamp(0, 1)
		nst(input_image)
		content_loss = style_loss = 0
		for item in content_losses:
			content_loss += content_weight * item.loss
		for item in style_losses:
			style_loss += style_weight * item.loss
		total_loss = style_loss + content_loss
		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()
		if step % (steps // numupdate) == 0:
			print("Step {}/{} - Total loss = {}, Style loss = {}, Content loss = {}".format(
				step, steps, total_loss, style_loss, content_loss))
			
		
		if step % (steps // numimg) == 0:
			if interp == True: 
				current_image = imfit(input_image.to(device).clone())
				frame = iter + (step // (steps // numimg))
				name = pathname + "/Temp{}-0.jpg".format(frame + 1)
				current_image = current_image.save(name)
				files.append(name)
		if step + 1 == steps :
			if interp == False:
				current_image = imfit(input_image.to(device).clone())
				name = pathname + "/Style{}{}-0.jpg".format(content_num, iter + 1)
				current_image = current_image.save(name)
				files.append(name)




			
	return files



def run_styles(temp_folder, styles, content) :
	out_files = []
	for i in range(len(content)) :
		img_for_dim = Image.open(content[i])
		ratio = img_for_dim.width / img_for_dim.height
		global imgWidth 
		imgWidth = 1000
		global imgHeight 
		imgHeight = int(imgWidth / ratio // 1)
		for j in range(len(styles)) :
			file = styles[j]
			style_image = imageLoader(file)
			content_image = imageLoader(content[i])
			input_image = content_image.clone().to(device)
			out = run_nst(content_image, style_image, input_image, j, i, temp_folder, False)
			out_files.extend(out)
	return out_files

def run_maps(temp_folder, files, width, height) :
	imgHeight = height
	imgWidth = width
	out_files = []
	for i in range(len(files)) :
		file = files[i]
		style_image = imageLoader(file)
		content_image = imageLoader(file)
		input_image = torch.randn((1,3,imgHeight,imgWidth))
		out = run_nst(content_image, style_image, input_image, i, temp_folder, False)
		out_files.extend(out)
	return out_files

def run_interp(temp, files) :
	new_list = []
	numfiles = len(files)
	for i in range(numfiles) :
		new_list.append(files[i])
		if i == numfiles - 1 :
			im1 = imageLoader(files[i])
			im2 = imageLoader(files[0])
		else :
			im1 = imageLoader(files[i])
			im2 = imageLoader(files[i + 1])
		content_image = im2.clone().to(device)
		input_image = im1.clone().to(device)
		imgnames = run_nst(content_image, content_image, input_image, i, temp, True)
		for name in imgnames :
			new_list.append(name)


	return new_list