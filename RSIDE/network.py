'''
The network architecture consists of four modules
an encoder (E)
a decoder (D)
a multi-scale feature fusion module (MFF)
a refinement module (R)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import resnet

class Encoder(nn.Module):

    '''
    Encoder - conv1(3, 64) - block1(64, 256) - block2(256,512) - block3(512,1024) - block4(1024,2048)
    '''

    def __init__(self, model, num_features=2048):
        super(Encoder, self).__init__()

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
    

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        block1 = self.layer1(x)
        block2 = self.layer2(block1)
        block3 = self.layer3(block2)
        block4 = self.layer4(block3)

        return block1, block2, block3, block4


class UpProjection(nn.Sequential):

    '''
    2x2 up sampling
    branch 1 - 5x5 convolution - relu - 3x3 convolution
    branch 2 - 5x5 convolution
    out - adding up 1 and 2 - relu
    '''

    def __init__(self, num_in_features, num_out_features):
        super(UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_in_features, num_out_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_out_features)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(num_out_features, num_out_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_out_features)

        self.conv3 = nn.Conv2d(num_in_features, num_out_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(num_out_features)

    
    def forward(self, x, size):

        x = F.interpolate(x, size=size, mode='bilinear')

        x_branch1 = self.conv1(x)
        x_branch1 = self.bn1(x_branch1)
        x_branch1 = self.relu(x_branch1)

        x_branch1 = self.conv2(x_branch1)
        x_branch1 = self.bn2(x_branch1)

        x_branch2 = self.conv3(x)
        x_branch2 = self.bn3(x_branch2)

        out = self.relu(x_branch1+x_branch2)

        return out


class Decoder(nn.Module):

    '''
    conv2(2048,1024) - up1(1024,512) - up2(512,256) - up3(256,128) - up4(128,64)
    '''

    def __init__(self, num_features=2048):
        super(Decoder, self).__init__()

        self.conv2 = nn.Conv2d(num_features, num_features//2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features//2)
        num_features = num_features//2

        self.up1 = UpProjection(num_features, num_features//2)
        num_features = num_features//2

        self.up2 = UpProjection(num_features, num_features//2)
        num_features = num_features//2

        self.up3 = UpProjection(num_features, num_features//2)
        num_features = num_features//2

        self.up4 = UpProjection(num_features, num_features//2)

    
    def forward(self, block1, block2, block3, block4):

        x = self.conv2(block4)
        x = self.bn(x)
        x = F.relu(x)

        size1 = [block3.size(2), block3.size(3)]
        x1 = self.up1(x, size1)

        size2 = [block2.size(2), block2.size(3)]
        x2 = self.up2(x1, size2)

        size3 = [block1.size(2), block1.size(3)]
        x3 = self.up3(x2, size3)

        size4 = [block1.size(2)*2, block1.size(3)*2]
        x4 = self.up4(x3, size4)

        return x4



class MFF(nn.Module):

    '''
    MFF module integrates four different scale features from the encoder
    using up-projection and channel-wise concatenation. 
    The output of four encoder blocks are upsample x2,4,8,16 respectively
    Then they are concatenated and transformed by a conv layer
    Output - 64 channels
    bk1_up(256,16) - bk2_up(512,16) - bk3_up(1024,16) - bk4_up(2048,16) - conv3(64,64)
    input_channels = [256, 512, 1024, 2048]
    num_features=64
    '''

    def __init__(self, input_channels, num_features=64):
        super(MFF, self).__init__()

        self.bk1_up = UpProjection(input_channels[0], num_out_features=16)
        self.bk2_up = UpProjection(input_channels[1], num_out_features=16)
        self.bk3_up = UpProjection(input_channels[2], num_out_features=16)
        self.bk4_up = UpProjection(input_channels[3], num_out_features=16)

        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)


    def forward(self, block1, block2, block3, block4, size):

        x_bk1 = self.bk1_up(block1, size)
        x_bk2 = self.bk2_up(block2, size)
        x_bk3 = self.bk3_up(block3, size)
        x_bk4 = self.bk4_up(block4, size)

        x = torch.cat((x_bk1, x_bk2, x_bk3, x_bk4), 1)
        x = self.conv3(x)
        x = self.bn(x)
        x = F.relu(x)

        return x


class Refinement(nn.Module):

    '''
    The feature from decoder and fused multi-scale features from 
    MFF are concatenated in their channels and fed to the refinement
    module having 3 convolutional layers
    conv4(128,128) - conv5(128,128) - conv6(128,1)
    conv6 without batch normalization and relu
    '''

    def __init__(self, input_channels, num_features=128):
        super(Refinement, self).__init__()

        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features)
        
        self.conv5 = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn5 = nn.BatchNorm2d(num_features)

        self.conv6 = nn.Conv2d(num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)


    def forward(self, x):

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)

        return x


class Model(nn.Module):

    def __init__(self, model, num_features=2048, input_channels=[256, 512, 1024, 2048]):
        super(Model, self).__init__()

        self.Encoder = Encoder(model, num_features)
        self.Decoder = Decoder(num_features)
        self.MFF = MFF(input_channels)
        self.Refinement = Refinement(input_channels)


    def forward(self, x):

        block1, block2, block3, block4 = self.Encoder(x)
        d_out = self.Decoder(block1, block2, block3, block4)
        size = [d_out.size(2), d_out.size(3)]
        mff_out = self.MFF(block1, block2, block3, block4, size)
        concat = torch.cat((d_out, mff_out), 1)
        out = self.Refinement(concat)

        return out



