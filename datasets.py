import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, img_path, label_path, mode='train'):
        super().__init__()
        self.img_list = self.get_images(img_path)
        self.attributes, self.label_list = self.get_labels(label_path)
        self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                ]
            )
        if mode == 'train':
            self.img_list = self.img_list[:150000]
            self.label_list = self.label_list[:150000]
        elif mode == 'test':
            self.img_list = self.img_list[150000:]
            self.label_list = self.label_list[150000:]
            

    def get_images(self, root):
        img_list = [os.path.join(root, name) for name in os.listdir(root)]

        return img_list

    def get_labels(self, root):
        with open(root, 'r') as f:
            lines = f.readlines()

        attributes = lines[1].split()

        data_label = []
        for i, line in enumerate(lines[2:]):
            split = line.split()
            data = [int(item.replace('-1', '0')) for item in split[1:]]
            data_label.append(data)

        return attributes, data_label


    def __getitem__(self, index):
        img = Image.open(self.img_list[index % len(self.img_list)]).convert('RGB')
        img = self.transform(img)
        label = torch.Tensor(self.label_list[index % len(self.label_list)])

        return img, label

    def __len__(self):
        return len(self.img_list)

if __name__ == '__main__':
    img_path = '/Kiwi/Data1/Dataset/celebA/img_align_celeba/'
    label_path = '/Kiwi/Data1/Dataset/celebA/list_attr_celeba.txt'

    a = ImageDataset(img_path, label_path)
    img, label = a[0]
    print(img.shape)
    print(label.shape)

