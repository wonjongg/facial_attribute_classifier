import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from webcari import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

'''
    Hyper-paramters
'''
img_path = '/Kiwi/Data1/Dataset/WebCaricature/WebCaricature_aligned'
label_path = '/Kiwi/Data1/Dataset/WebCariA_Dataset/WebCariA_Dataset/all_photo_data.txt'
#model_path = 'checkpoint/photo_99.pth'
train_epochs = 100
device = 'cuda'

model = ResNet18().to(device)

dataset = ImageDataset(img_path, label_path, 'train')
trainloader = DataLoader(dataset, batch_size=64, shuffle=True)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
model.train()

for epoch in range(train_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        images, labels = data[0], data[1]
        images, labels = images.to(device), labels.to(device) 

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        print(f'[{epoch+1}, {i+1}] loss: {running_loss/100:.3f}')
        running_loss = 0.0

    scheduler.step()
    torch.save(model.state_dict(), f'checkpoint/photo_res18_{epoch}.pth')
