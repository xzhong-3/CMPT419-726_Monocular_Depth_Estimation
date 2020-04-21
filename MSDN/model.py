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

class GlobalCoarseNet(nn.Module):   
    def __init__(self, init=True):
        super(GlobalCoarseNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4), 
                                     #nn.BatchNorm2d(96),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), 
                                     #nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2))
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), 
                                     #nn.BatchNorm2d(384),
                                     nn.ReLU())
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), 
                                     #nn.BatchNorm2d(384),
                                     nn.ReLU())
        
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=2), 
                                     #nn.BatchNorm2d(256),
                                     nn.ReLU())
        
        self.fc1 = nn.Sequential(nn.Linear(in_features=256 * 8 * 6, out_features=4096), 
                                     nn.ReLU(), nn.Dropout(0.5))
        
        self.fc2 = nn.Sequential(nn.Linear(in_features=4096, out_features=74 * 55))
        
        if init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.reshape(x.size(0), 55, 74)
        return x

    #     self.conv1 = nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 0)
    #     self.conv2 = nn.Conv2d(96, 256, kernel_size = 5, padding = 2)
    #     self.conv3 = nn.Conv2d(256, 384, kernel_size = 3, padding = 1)
    #     self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, padding = 1)
    #     self.conv5 = nn.Conv2d(384, 256, kernel_size = 3, stride = 2)
    #     self.fc1 = nn.Linear(12288, 4096)
    #     self.fc2 = nn.Linear(4096, 4070)
    #     self.pool = nn.MaxPool2d(2)
    #     self.dropout = nn.Dropout2d()
        
    #     if init:
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 m.weight.data.normal_(0, 0.01)
    #                 if m.bias is not None:
    #                     m.bias.data.zero_()
    #                 elif isinstance(m, nn.Linear):
    #                     m.weight.data.normal_(0, 0.01)
    #                     m.bias.data.zero_()

    # def forward(self, x):
    #     x = self.conv1(x)                       # [32, 96, 74, 55]
    #     x = F.relu(x)
    #     x = self.pool(x)                        # [32, 96, 37, 27]
    #     x = self.conv2(x)                       # [32, 256, 33, 23]
    #     x = F.relu(x)
    #     x = self.pool(x)                        # [32, 256, 16, 11] 18X13
    #     x = self.conv3(x)                       # [32, 384, 14, 9]
    #     x = F.relu(x)
    #     x = self.conv4(x)                       # [32, 384, 12, 7]
    #     x = F.relu(x)
    #     x = self.conv5(x)                       # [32, 256, 10, 5] 8X5
    #     x = F.relu(x)
    #     x = x.view(x.size(0), -1)               # [32, 12800]
    #     x = F.relu(self.fc1(x))                 # [32, 4096]
    #     x = self.dropout(x)
    #     x = self.fc2(x)                         # [32, 4070]     => 55x74 = 4070
    #     x = x.view(x.size(0), 74, 55)           # [32, 74, 55]
    #     return x

class LocalFineNet(nn.Module): 
    def __init__(self, init=True):
        super(LocalFineNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 63, kernel_size = 9, stride = 2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv2d(64, 1, kernel_size = 5, padding = 2)
        self.pool = nn.MaxPool2d(2)
        
        if init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()
                    elif isinstance(m, nn.Linear):
                        m.weight.data.normal_(0, 0.01)
                        m.bias.data.zero_()

    def forward(self, x, global_output_batch):
        x = F.relu(self.conv1(x))                     # [32, 63, 148, 110]
        x = self.pool(x)                              # [32, 63, 74, 55]
        x = torch.cat((x,global_output_batch),1)      # x = [8, 63, 74, 55], y = [8, 1, 74, 55] => x = [8, 64, 74, 55]
        x = F.relu(self.conv2(x))                     # [32, 64, 74, 55]
        x = self.conv3(x)                             # [32, 1, 74, 55]  
        return x