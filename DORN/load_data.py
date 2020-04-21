import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import h5py
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


data_path = './data/nyu_depth_v2_labeled.mat'
batch_size = 2
iheight, iwidth = 480, 640 # raw image size
alpha, beta = 0.02, 10.02
K = 68
output_size = (257, 353)

class NYU_Depth_v2(data.Dataset):
    def __init__(self, data_path, lists):
        self.data_path = data_path
        self.lists = lists
        self.nyu = h5py.File(self.data_path)
        self.imgs = self.nyu['images']
        self.dpts = self.nyu['depths']
        self.output_size = (257, 353)
    
    def __getitem__(self, index):
        img_idx = self.lists[index]
        img = self.imgs[img_idx].transpose(2, 1, 0) #HWC
        dpt = self.dpts[img_idx].transpose(1, 0)
        img = Image.fromarray(img)
        dpt = Image.fromarray(dpt)
        img_transform = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(self.output_size),
            transforms.ToTensor()
        ])
        dpt_transform = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(self.output_size),
            transforms.ToTensor()
        ])
        img = img_transform(img)
        dpt = dpt_transform(dpt)
        dpt = get_depth_log(dpt)
        return img, dpt

    def __len__(self):
        return len(self.lists)

def get_depth_log(depth):
    """
    Perform SID based on its depth label
    """
    alpha_ = torch.FloatTensor([alpha])
    beta_ = torch.FloatTensor([beta])
    K_ = torch.FloatTensor([K])
    t = K_ * torch.log(depth / alpha_) / torch.log(beta_ / alpha_)
    # t = t.int()
    return t


def get_depth_sid(depth_labels):
    """
    Get Original Depth based on SID label
    """
    depth_labels = depth_labels.data.cpu()
    alpha_ = torch.FloatTensor([alpha])
    beta_ = torch.FloatTensor([beta])
    K_ = torch.FloatTensor([K])
    t = torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * depth_labels / K_)
    return t

def getNYUDataset():
    train_lists, val_lists, test_lists = load_split()
    train = NYU_Depth_v2(data_path=data_path, lists = range(0, 1000))
    train_loader = data.DataLoader(train, batch_size = batch_size, shuffle = True, drop_last=True)

    test = NYU_Depth_v2(data_path=data_path, lists = range(1000,:))
    test_loader = data.DataLoader(test, batch_size = 1, shuffle = False, drop_last=True)

    val_set = NYU_Dataset(data_path=data_path, lists=val_lists)
    val_loader = data.DataLoader(val_set, batch_size=1, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader

def load_split():
    current_directoty = os.getcwd()
    train_lists_path = current_directoty + '/data/trainIdxs.txt'
    test_lists_path = current_directoty + '/data/testIdxs.txt'

    train_f = open(train_lists_path)
    test_f = open(test_lists_path)

    train_lists = []
    test_lists = []

    train_lists_line = train_f.readline()
    while train_lists_line:
        train_lists.append(int(train_lists_line) - 1)
        train_lists_line = train_f.readline()
    train_f.close()

    test_lists_line = test_f.readline()
    while test_lists_line:
        test_lists.append(int(test_lists_line) - 1)
        test_lists_line = test_f.readline()
    test_f.close()

    val_start_idx = int(len(train_lists) * 0.8)

    val_lists = train_lists[val_start_idx:-1]
    train_lists = train_lists[0:val_start_idx]

    return train_lists, val_lists, test_lists

if __name__ == '__main__':
    load_test()
