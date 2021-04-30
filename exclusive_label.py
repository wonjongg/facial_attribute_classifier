import os

import numpy as np

attribute_path = '/Kiwi/Data1/Dataset/WebCariA_Dataset/WebCariA_Dataset/labels.txt'
with open(attribute_path, 'r') as f:
    attributes = [l.rstrip() for l in f.readlines()]

path = '/Kiwi/Data1/Dataset/WebCariA_Dataset/WebCariA_Dataset/all_cari_data.txt'
with open(path, 'r') as f:
    lines = f.readlines()

labels = np.zeros((len(lines), 50))
id = []
for i, label in enumerate(labels):
    labels[i] = np.array([int(item) for item in lines[i].split()[1:]])
    id.append(lines[i].split()[0].split('C')[0])

key = id[0]
sum_label = np.zeros(50)
for i in range(len(id)):
    if not key == id[i]:
        break
    sum_label += labels[i]

print(i + 1)
for i, label in enumerate(sum_label):
    print(f'[{i}] {label}')

for i, att in enumerate(attributes):
    print(f'[{i}] {att}')

sum_labelp = sum_label >= 20

attribute_path = '/Kiwi/Data1/Dataset/WebCariA_Dataset/WebCariA_Dataset/labels.txt'
with open(attribute_path, 'r') as f:
    attributes = [l.rstrip() for l in f.readlines()]

path = '/Kiwi/Data1/Dataset/WebCariA_Dataset/WebCariA_Dataset/all_photo_data.txt'
with open(path, 'r') as f:
    lines = f.readlines()

labels = np.zeros((len(lines), 50))
id = []
for i, label in enumerate(labels):
    labels[i] = np.array([int(item) for item in lines[i].split()[1:]])
    id.append(lines[i].split()[0].split('P')[0])

key = id[0]
sum_label = np.zeros(50)
for i in range(len(id)):
    if not key == id[i]:
        break
    sum_label += labels[i]

print(i + 1)
for i, label in enumerate(sum_label):
    print(f'[{i}] {label}')

for i, att in enumerate(attributes):
    print(f'[{i}] {att}')

sum_labelc = sum_label >= 5
print(sum_labelp)
print(sum_labelc)

sum_label = np.logical_xor(sum_labelp, sum_labelc)
for i, label in enumerate(sum_label):
    print(f'[{i}] {label}')
