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
from dataset import NYUDataset
from model import GlobalCoarseNet, LocalFineNet
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings('ignore')

def scale_invariant_loss(output, target):
    mask = (target == 0) | (target == target.max()) | (target == target.min())
    di = output[~mask] - torch.log(target[~mask])
    di = output - target
    loss = torch.mean(di ** 2) - 0.5 / ((torch.numel(di)) ** 2) * (torch.sum(di) ** 2) 

    return loss

# train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
filename = '../data/nyu_depth_v2_labeled.mat'
# f = h5py.File(filename, 'r')
bs = 32 # batch size
train_transform = transforms.Compose([
                                # transforms.Resize((304, 228)),
                                # transforms.RandomRotation(degrees=(-5,5)), # doesnt work
                                # transforms.RandomCrop((304, 228)),
                                transforms.RandomCrop((228, 304)),
                                transforms.RandomHorizontalFlip(),
                                transforms.Resize((228, 304)),
                            ])
test_transform = transforms.Compose([
                                # transforms.Resize((304, 228)),
                                transforms.Resize((228, 304)),
                            ])
train_loader = DataLoader(NYUDataset( filename,
                                        'training', 
                                        transform = train_transform), 
                                        batch_size = bs,
                                        drop_last = True, 
                                        shuffle = True)
validation_loader = DataLoader(NYUDataset( filename,
                                            'validation', 
                                            transform = test_transform), 
                                            batch_size = bs, 
                                            drop_last = True, 
                                            shuffle = False)

test_loader = DataLoader(NYUDataset( filename,
                                        'test', 
                                        transform = test_transform), 
                                        batch_size = 1, 
                                        drop_last = True, 
                                        shuffle = False)

epochs = 10
global_model = GlobalCoarseNet(init=False).to(device)
local_model = LocalFineNet(init=False).to(device)

global_optimizer = torch.optim.SGD([{'params': global_model.fc1.parameters(), 'lr': 0.01},
                                    {'params': global_model.fc2.parameters(), 'lr': 0.01}], 
                                   lr=0.0001, momentum=0.9, weight_decay=0.1)

local_optimizer = torch.optim.SGD([{'params': local_model.conv2.parameters(), 'lr': 0.001}], 
                                  lr=0.0001, momentum=0.9, weight_decay=0.1)

print("Start training global model:")
# global model
for epoch in range(epochs):
    # train
    train_loss = 0
    global_model.train()
    for i, samples in enumerate(train_loader):
        images = samples['image'].float().to(device)
        depths = samples['depth'].float().to(device)
        
        # forward pass
        output = global_model(images)
        loss = scale_invariant_loss(output, depths)

        # backward pass
        global_model.zero_grad()
        loss.backward()

        # optimization
        global_optimizer.step()
        
        train_loss += loss.item()
        
    # validation
    valid_loss = 0
    global_model.eval()
    with torch.no_grad():
        for i, samples in enumerate(validation_loader):
            images = samples['image'].float().to(device)
            depths = samples['depth'].float().to(device)

            # forward pass
            output = global_model(images)
            loss = scale_invariant_loss(output, depths)
            
            valid_loss += loss.item()
            
    print(('> epoch {epoch} done, train_loss = {tloss}, validation loss = {vloss}').format(epoch=epoch, tloss=train_loss, vloss=valid_loss))
# save model
# torch.save(global_model, './models/global_model.pt')

print("global model training finish")
print()
print("start training local model:")

local_train_loss = []
for epoch in range(epochs):
    # train
    train_loss = 0
    local_model.train()
    global_model.eval()
    for i, samples in enumerate(train_loader):
        images = samples['image'].float().to(device)
        depths = samples['depth'].float().to(device)
        # images = samples['image'].float().cuda()
        # depths = samples['depth'].float().cuda()
        
        # forward pass
        global_output_batch = global_model(images)
        global_output_batch = global_output_batch.unsqueeze(1)
        output = local_model(images, global_output_batch)
        output = output.squeeze()
        loss = scale_invariant_loss(output, depths)

        # backward pass
        local_model.zero_grad()
        loss.backward()

        # optimization
        local_optimizer.step()

        train_loss += loss.item()
    
    # validation
    valid_loss = 0
    local_model.eval()
    with torch.no_grad():
        for i, samples in enumerate(validation_loader):
            images = samples['image'].float().to(device)
            depths = samples['depth'].float().to(device)

            # forward pass
            output = local_model(images, global_output_batch)
            output = output.squeeze()
            loss = scale_invariant_loss(output, depths)
            
            valid_loss += loss.item()
                
    print(('> epoch {epoch} done, train_loss = {tloss}, validation loss = {vloss}').format(epoch=epoch, tloss=train_loss, vloss=valid_loss))
# # save model
# torch.save(local_model, '/content/drive/My Drive/CMPT419/Project/model/local_model.pt')

print("local model training finish")

