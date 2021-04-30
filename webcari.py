import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, webcari_path, label_path, mode='train'):
        super().__init__()
        self.path = webcari_path
        self.id_list = os.listdir(self.path)
        self.data = self.get_labels(label_path)
        if mode == 'train' or mode == 'all':
            self.transform = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                    ]
                )
        else:
            self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                    ]
                )

        random.seed(4322)

        self.len = len(self.data['filename'])
        index_range = range(0, self.len)
        self.test_index = random.sample(index_range, (self.len) // 10)
        self.train_index = list(set(list(index_range)) - set(self.test_index))
        self.mode = mode

    def get_labels(self, root):
        with open(root, 'r') as f:
            lines = f.readlines()

        data = {'name':[], 'filename': [], 'label': []}
        for i, line in enumerate(lines):
            split = line.split()
            name_split = split[0].split('_')
            if len(name_split) > 2:
                name_split[0] = '-'.join(name_split[:-1])
            name = name_split[0].replace('-', '_')
            key1 = name.split('_')[0]
            key2 = name.split('_')[-1]
            name = [id for id in self.id_list if id.startswith(key1) and id.endswith(key2)][0]
            filename = name_split[-1] + '.jpg'

            label = [int(item) for item in split[1:]]

            data['name'].append(name)
            data['filename'].append(filename)
            data['label'].append(label)

        return data


    def __getitem__(self, index):
        if self.mode == 'train':
            idx = index % len(self.train_index)
            idx = self.train_index[idx]

        elif self.mode == 'test':
            idx = index % len(self.test_index)
            idx = self.test_index[idx]

        else:
            idx = index % self.len

        name = os.path.join(self.path, self.data['name'][idx])
        filepath = os.path.join(name, self.data['filename'][idx])

        img = Image.open(filepath).convert('RGB')
        img = self.transform(img)
        label = torch.Tensor(self.data['label'][idx])

        return img, label

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_index)
        elif self.mode == 'test':
            return len(self.test_index)
        else:
            return self.len

if __name__ == '__main__':
    img_path = '/Kiwi/Data1/Dataset/WebCaricature/WebCaricature_aligned'
    label_path = '/Kiwi/Data1/Dataset/WebCariA_Dataset/WebCariA_Dataset/all_cari_data.txt'

    a = ImageDataset(img_path, label_path, 'train')
    img, label = a[0]
    print(img.shape)
    print(label.shape)

