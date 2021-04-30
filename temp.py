import os
import numpy as np

path = '/Kiwi/Data1/Dataset/WebCariA_Dataset/WebCariA_Dataset/all_cari_data.txt'

with open(path, 'r') as f:
    lines = f.readlines()

attr_path = '/Kiwi/Data1/Dataset/WebCariA_Dataset/WebCariA_Dataset/labels.txt'

with open(attr_path, 'r') as f:
    attrs = f.readlines()

attrs = [attr.rstrip() for attr in attrs]

ret = np.zeros(50)
for line in lines:
    split = line.split()
    split = [int(item) for item in split[1:]]
    ret += np.array(split)

ret = 100 * ret / len(lines)

for i, val in enumerate(ret):
    print(f'{attrs[i]}: {val:.1f}')
