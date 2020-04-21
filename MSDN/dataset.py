import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, utils
from skimage import transform
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import time
import os
import re
import random
import h5py
from imageio import imread
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings('ignore')

class NYUDataset(Dataset):
    def __init__(self, filename, type, transform = None):       
        f = h5py.File(filename, 'r')
        # f = file
        images_data = f['images']
        depths_data = f['depths']
        # images_data = np.transpose(images_data, (0,3,2,1))
        # depths_data = np.transpose(depths_data, (0,2,1))
        if type == "training":
            self.images = images_data[0:1024]
            self.depths = depths_data[0:1024]
        elif type == "validation":
            self.images = images_data[1024:1248]
            self.depths = depths_data[1024:1248]
        elif type == "test":
            self.images = images_data[1248:]
            self.depths = depths_data[1248:]

        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):  
        image = self.images[idx]
        image = np.transpose(image, (1,2,0))
        # print('image:', type(image))
        # input('...')
        image = Image.fromarray(image)
        # print('image:',image)
        # input('...')

        depth = self.depths[idx]
        # depth = np.transpose(depth, (1,0))
        depth = Image.fromarray(depth)
        # print('depth:',depth)
        # input('...')
        # depth = np.reshape(depth, (1, depth.shape[0], depth.shape[1]))

        seed = random.randint(0, 2147483647)
        if self.transform:
            random.seed(seed)
            image = self.transform(image)
            random.seed(seed)
            depth = self.transform(depth)

        # convert to torch tensor
        image = transforms.ToTensor()(image)
        # NYU dataset statistics
        images_mean = [109.31410628 / 255, 109.31410628 / 255, 109.31410628 / 255]
        images_std = [76.18328376 / 255, 76.18328376 / 255, 76.18328376 / 255]
        image = transforms.Normalize(images_mean, images_std)(image)
        depth = transforms.Resize((55, 74))(depth)
        # print('depth', depth)
        depth = transforms.ToTensor()(depth).view(55, 74)
        # print('depth', depth)
        # print(depth.shape)
        # input('...')
        
        sample = {'image': image, 'depth': depth}
        return sample
