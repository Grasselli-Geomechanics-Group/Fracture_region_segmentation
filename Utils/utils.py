import numpy as np
from itertools import product
import os
from PIL import Image
import random


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def insert(self, key, value):
        self[key] = value


def crop2patch(img_name, dim, flag_is_x):
    img = Image.open(img_name)
    # pad img
    col, row = img.size    
    pad_c = dim-col%dim
    pad_r = dim-row%dim
    if flag_is_x:
        img_pad = np.pad(img, [(0, pad_r), (0, pad_c)], mode='constant', constant_values=0)
    else:
        img_pad = np.pad(img, [(0, pad_r), (0, pad_c)], mode='constant', constant_values=1)

    img_patches = []
    grid = list(product(range(0, row+pad_r, dim), range(0, col+pad_c, dim)))
    for i, j in grid:
        img_patch = img_pad[i:i+dim, j:j+dim]
        img_patch = img_patch[np.newaxis, :, :]
        img_patch = img_patch/255 if flag_is_x else (img_patch - 1) ## -1 to make labels from 1/2/3 to 0/1/2
        img_patches.append(img_patch)
    return img_patches, img.size, img_pad.shape


def getFilesInPath(folder, suffix, contain_t):
    name_list = []
    f_list = sorted(os.listdir(folder))
    for f_n in f_list:
        if contain_t is not None:
            if suffix in os.path.splitext(f_n)[1] and contain_t in os.path.splitext(f_n)[0]:
                pathName = os.path.join(folder, f_n)
                name_list.append(pathName)
        else:
            if suffix in os.path.splitext(f_n)[1]:
                pathName = os.path.join(folder, f_n)
                name_list.append(pathName)

    return name_list


def getFilesInSubPath(model_to_load, folder, suffix, contain_t):
    name_list = []
    f_list = sorted(os.listdir(folder))
    folder_sub = [f_n for f_n in f_list if model_to_load in f_n][0] #time stamp is unique, for sure
    folder_sub = os.path.join(folder, folder_sub)
    f_sub_list = sorted(os.listdir(folder_sub))

    for f_n in f_sub_list:
        if contain_t is not None:
            if suffix in os.path.splitext(f_n)[1] and contain_t in os.path.splitext(f_n)[0]:
                pathName = os.path.join(folder_sub, f_n)
                name_list.append(pathName)
        else:
            if suffix in os.path.splitext(f_n)[1]:
                pathName = os.path.join(folder_sub, f_n)
                name_list.append(pathName)

    return name_list