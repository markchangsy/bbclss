import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
	def __init__(self, dataset, pretrained=True):
		super(ResNet, self).__init__()
		num_classes = 50 if dataset=="ESC" else 10
		self.model = models.resnet50(pretrained=pretrained)
		self.model.fc = nn.Linear(2048, num_classes)
		
	def forward(self, x):
		output = self.model(x)
		return output

class ResNet34(nn.Module):
        def __init__(self, dataset, pretrained=True):
                super(ResNet34, self).__init__()
                # num_classes = 50 if dataset=="ESC" else 12
                num_classes = 13
                # num_classes = 2
                self.model = models.resnet34(pretrained=pretrained)
                # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                self.model.fc = nn.Linear(512, num_classes)
		
        def forward(self, x):
                output = self.model(x)
                return output

class ResNet18(nn.Module):
        def __init__(self, dataset, pretrained=True):
                super(ResNet18, self).__init__()
                num_classes = 50 if dataset=="ESC" else 10
                self.model = models.resnet34(pretrained=False)
                # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                self.model.fc = nn.Linear(512, num_classes)
		
        def forward(self, x):
                output = self.model(x)
                return output

class ResNet50(nn.Module):
        def __init__(self, dataset, pretrained=True):
                super(ResNet50, self).__init__()
                # num_classes = 50 if dataset=="ESC" else 12
                num_classes = 12
                # num_classes = 2
                self.model = models.resnet50(pretrained=pretrained)
                # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                self.model.fc = nn.Linear(2048, num_classes)
		
        def forward(self, x):
                output = self.model(x)
                return output

class EfficientNet(nn.Module):
        def __init__(self, dataset, pretrained=True):
                super(EfficientNet, self).__init__()
                # num_classes = 50 if dataset=="ESC" else 12
                num_classes = 12
                # num_classes = 2
                self.model = models.efficientnet_b3(pretrained=pretrained)
                # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                self.model.fc = nn.Linear(1280, num_classes)
		
        def forward(self, x):
                output = self.model(x)
                return output

class AVENet(nn.Module):

    def __init__(self, dataset):
        super(AVENet, self).__init__()
        num_classes = 50 if dataset=="ESC" else 10
        self.audnet = models.resnet18(pretrained=False)
        self.audnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.audnet.fc = nn.Linear(512, num_classes)

    def forward(self, audio):
        aud = self.audnet(audio)
        return aud