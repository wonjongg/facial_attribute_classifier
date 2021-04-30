import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from torchvision.models import vgg16, resnet18, resnet50

import sys
import math

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg = vgg16(pretrained=True)
#        self.vgg.fc = nn.Linear(2048, 50)

    def forward(self, x):
        x = self.vgg(x)
        return torch.sigmoid(x)

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 50)

    def forward(self, x):
        x = self.resnet(x)
        return F.sigmoid(x)

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048, 50)

    def forward(self, x):
        x = self.resnet(x)
        return F.sigmoid(x)

class WideResNet50_2(nn.Module):
    def __init__(self):
        super().__init__()

        self.wresnet = wide_resnet50_2(pretrained=True)
        self.wresnet.fc = nn.Linear(2048, 50)

    def forward(self, x):
        x = self.wresnet(x)
        return F.sigmoid(x)


if __name__ == '__main__':
    print(resnet18(pretrained=True))
