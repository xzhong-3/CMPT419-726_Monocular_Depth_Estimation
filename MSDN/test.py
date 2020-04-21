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
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# test
def compute_err(predict, gt):
    threshold = np.maximum((gt / predict), (predict / gt))
    threshold1 = (threshold < 1.25).mean()
    threshold2 = (threshold < 1.25 ** 2).mean()
    threshold3 = (threshold < 1.25 ** 3).mean()

    rmse = (gt - predict) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(predict)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - predict) / gt)

    sq_rel = np.mean(((gt - predict)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, threshold1, threshold2, threshold3

def scale_invariant_loss(output, target):
    di = output - target
    loss = torch.mean(di ** 2) - 0.5 / (torch.numel(di)**2) * (torch.sum(di) ** 2)
    return loss

global_model = torch.load('/content/drive/My Drive/CMPT419/Project/model/global_model.pt')
local_model = torch.load('/content/drive/My Drive/CMPT419/Project/model/local_model.pt')
global_model.eval()
local_model.eval()
size = len(test_loader.dataset)
avg_abs_rel = 0 
avg_sq_rel = 0 
avg_rmse_linear = 0
avg_rmse_log = 0
avg_rmse_scale = 0
t1 = 0
t2 = 0
t3 = 0
for i, samples in enumerate(test_loader):
    images = samples['image'].float().to(device)
    depths = samples['depth'].float().to(device)
    global_output_batch = global_model(images)
    global_output_batch = global_output_batch.unsqueeze(1)
    output = local_model(images, global_output_batch)
    output = output.squeeze()

    abs_rel, sq_rel, rmse, rmse_log, threshold1, threshold2, threshold3 = compute_err(output.cpu().detach().numpy(), depths.cpu().detach().numpy())
    rmse_scale = scale_invariant_loss(depths, output)
    avg_abs_rel += abs_rel / size
    avg_sq_rel += sq_rel / size
    avg_rmse_linear += rmse / size
    avg_rmse_log += rmse_log / size
    avg_rmse_scale += rmse_scale / size
    t1 += threshold1 / size
    t2 += threshold2 / size
    t3 += threshold3 / size
    if i == 0:
        plt.figure()
        depth_img = depths.cpu().detach().numpy().reshape((55, 74)) 
        depth_img = depth_img / 10 * 255
        plt.imshow(depth_img, cmap='gray', vmin=0, vmax=255)
        plt.figure()
        output_img = output.cpu().detach().numpy().reshape((55,74))
        output_img = output_img / 10 * 255
        plt.imshow(output_img, cmap='gray', vmin=0, vmax=255)

print('error (threshold < 1.25):', t1)
print('error (threshold < 1.25**2):', t2)
print('error (threshold < 1.25**3):', t3)
print('error (abs relative difference):', avg_abs_rel)
print('error (sqr relative difference):', avg_sq_rel)
print('error (RMSE linear):', avg_rmse_linear)
print('error (RMSE log):', avg_rmse_log)
print('error (RMSE log scale):', avg_rmse_scale)