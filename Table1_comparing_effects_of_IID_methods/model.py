import torch
from torchvision.models import resnet18, vgg19_bn, resnet50, alexnet
from torch.autograd import Variable
import torch.nn.functional as F
import random
import math
from inceptionResnetV2 import inceptionresnetv2

class MyAlexnet(torch.nn.Module):

	def __init__(self):
		super(MyAlexnet, self).__init__()

		self.pretrained = alexnet(pretrained=True)
		self.pretrained._modules['classifier'] = self.pretrained._modules['classifier'][:-1]
		self.net = self.pretrained

		self.class_classifier = torch.nn.Linear(4096, 7)

	def forward(self, x):
		x = self.pretrained(x)
		x = self.class_classifier(x)
		return x;

class MyInceptionResnet(torch.nn.Module):

	def __init__(self):
		super(MyInceptionResnet, self).__init__()

		self.pretrained = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
		self.layers = list(self.pretrained._modules.keys())
		self.pretrained._modules.pop(self.layers[-1])
		self.pretrained._modules.pop(self.layers[-2])
		self.net = torch.nn.Sequential(self.pretrained._modules)
		self.pretrained = None

		self.avgpool = torch.nn.AvgPool2d(5, stride=1)
		self.class_classifier = torch.nn.Linear(1536, 7)

	def forward(self, x):
		x = self.net(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.class_classifier(x)
		return x;

class MyResnet(torch.nn.Module):

	def __init__(self):
		super(MyResnet, self).__init__()

		self.pretrained = resnet18(pretrained=True)
		self.layers = list(self.pretrained._modules.keys())
		self.pretrained._modules.pop(self.layers[-1])
		self.net = torch.nn.Sequential(self.pretrained._modules)
		self.pretrained = None

		self.avgpool = torch.nn.AvgPool2d(7, stride=1)
		self.class_classifier = torch.nn.Linear(512, 7)


	def forward(self, x):
		x = self.net(x)
		x = x.view(x.size(0), -1)
		x = self.class_classifier(x)
		return x

class MyVGG(torch.nn.Module):

	def __init__(self):
		super(MyVGG, self).__init__()
		self.model = vgg19_bn(pretrained=True)
		self.model.classifier = self.model.classifier[:-1]
		self.class_classifier = torch.nn.Linear(4096, 7)

	def forward(self, x):
		x = self.model(x)
		x = self.class_classifier(x)
		return x
