import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import *
from webcari import *

from torch.utils.data import DataLoader

import os
import numpy as np

from PIL import Image

img_path = '/Kiwi/Data1/Dataset/WebCaricature/WebCaricature_aligned'
label_path = '/Kiwi/Data1/Dataset/WebCariA_Dataset/WebCariA_Dataset/all_photo_data.txt'
label_name_path = '/Kiwi/Data1/Dataset/WebCariA_Dataset/WebCariA_Dataset/labels.txt'
model_path = 'checkpoint/photo_res18_99.pth'
device ='cuda'

with open(label_name_path, 'r') as f:
    lines = f.readlines()
    attributes = [line.rstrip() for line in lines]


model = ResNet18().to(device)
model.load_state_dict(torch.load(model_path))

model.eval()

image = Image.open('test0.png').convert('RGB')
image = np.array(image)
image = torch.Tensor(2 * (image / 255 - 0.5)).permute(2, 0, 1)
print(image)

image = image.to(device).unsqueeze(0)

predicted = model(image).squeeze()
predicted[predicted > 0.5] = 1
predicted[predicted <= 0.5] = 0

for i, item in enumerate(attributes):
    print(f'{item}: {predicted[i]}')
