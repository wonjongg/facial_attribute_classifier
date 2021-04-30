"""Align face images given landmarks."""

# MIT License
# 
# Copyright (c) 2017 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from align.matlab_cp2tform import get_similarity_transform_for_cv2
from matlab_cp2tform import get_similarity_transform_for_cv2

import numpy as np
from scipy import misc
import sys
import os
import argparse
import random
import cv2
import matplotlib.pyplot as plt

def align(src_img, src_pts, ref_pts, image_size, scale=1.0, transpose_input=False):
    w, h = image_size = tuple(image_size)

    # Actual offset = new center - old center (scaled)
    scale_ = max(w,h) * scale
    cx_ref = cy_ref = 0.
    offset_x = 0.5 * w - cx_ref * scale_
    offset_y = 0.5 * h - cy_ref * scale_

    s = np.array(src_pts).astype(np.float32).reshape([-1,2])
    r = np.array(ref_pts).astype(np.float32) * scale_ + np.array([[offset_x, offset_y]])
    if transpose_input: 
        s = s.reshape([2,-1]).T

    tfm = get_similarity_transform_for_cv2(s, r)
    dst_img = cv2.warpAffine(src_img, tfm, image_size)

    s_new = np.concatenate([s.reshape([2,-1]), np.ones((1, s.shape[0]))])
    s_new = np.matmul(tfm, s_new)
    s_new = s_new.reshape([-1]) if transpose_input else s_new.T.reshape([-1]) 
    tfm = tfm.reshape([-1])
    return dst_img, s_new, tfm


def main(args):
    dir_name = '/Kiwi/Data1/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebA_resized'
    target_dir = '/Kiwi/Data1/Dataset/CelebAMask-HQ/CelebAMask-HQ/align'
    input_file = '/Kiwi/Data1/Dataset/celebA/list_landmarks_align_celeba.txt'

    with open(input_file, 'r') as f:
        lines = f.readlines()

    ref_pts = np.array( [[ -1.58083929e-01, -3.84258929e-02],
                         [  1.56533929e-01, -4.01660714e-02],
                         [  2.25000000e-04,  1.40505357e-01],
                         [ -1.29024107e-01,  3.24691964e-01],
                         [  1.31516964e-01,  3.23250893e-01]])

    src_pts_list = [[float(item) for item in line.strip().split()[1:]] for line in lines[2:]]

    mapping_file = '/Kiwi/Data1/Dataset/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt'

    with open(mapping_file, 'r') as f:
        lines = f.readlines()

    mapping = [line.split()[1] for line in lines[1:]]

    for i, fname in enumerate(os.listdir(dir_name)):
        idx = int(fname.split('/')[-1].replace('.jpg', '')) - 1
        fpath = os.path.join(dir_name, fname)
        src_pts = src_pts_list[int(mapping[idx])]
        for j in range(len(src_pts)):
            src_pts[j] = src_pts[j] * 99/69 if j % 2 else src_pts[j] * 123 / 110
        img = misc.imread(fpath)
        img_new, new_pts, tfm = align(img, src_pts, ref_pts, args.image_size, args.scale, args.transpose_input)
        print(src_pts)

        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        img_path_new = os.path.join(target_dir, fname)
        misc.imsave(img_path_new, img_new)
        if i % 100==0:
            print(img_path_new)

    return


        

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, nargs=2,
        help='Image size (height, width) in pixels.', default=[256, 256])
    parser.add_argument('--scale', type=float,
        help='Scale the face size in the target image.', default=1.0)
    parser.add_argument('--dir_depth', type=int,
        help='When writing into new directory, how many layers of the dir tree should be kept.', default=2)
    parser.add_argument('--transpose_input', action='store_true',
        help='Set true if the input landmarks is in the format x1 x2 ... y1 y2 ...')
    parser.add_argument('--visualize', action='store_true',
        help='Visualize the aligned images.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
