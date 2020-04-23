import os
import torch
import torch.nn as nn
import torchvision.models
import collections
import math

def weights_int(modules):
    # Initialize filters using Gaussian random weights
    if isinstance(modules, nn.Conv2d):
        n = modules.kernel_size[0] * modules.kernel_size[1] * modules.out_channels
        modules.weight.data.normal_(0, math.sqrt(2. / n))
        if modules.bias is not None:
            modules.bias.data.zero_()
    elif isinstance(modules, nn.ConvTranspose2d):
        n = modules.kernel_size[0] * modules.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if modules.bias is not None:
            modules.bias.data.zero_()
    elif isinstance(modules, nn.BatchNorm2d):
        modules.weight.data.fill_(1)
        modules.bias.zero_()

def conv3x3(inplanes, outplanes, stride = 1):
    "3x3 convolution with padding"
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride = stride, padding = 1, bias = False)

def conv1x1(inplanes, outplanes, stride = 1):
    "1x1 convolution"
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, stride = stride, bias = False)

## The Dense Feature Extractor is a regular DCNN
## It is mentioned in the paper that they used a ResNet-101 as the Dense Feature Extractor

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, dilation = 1, downsample = None, multi_grid=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride = stride, 
                                padding = dilation * multi_grid, dilation = dilation * multi_grid, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.dilation = dilation
        self.stride = stride
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class DenseFeatureExtractor(nn.Module):
    
    def __init__(self, block, layers):
        super(DenseFeatureExtractor, self).__init__()
        self.inplanes = 128
        self.conv1 = conv3x3(3, 64, stride = 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(64,64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1, ceil_mode = True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 1, dilation = 4, multi_grid = (1,1,1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation = 1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid = generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation = dilation, multi_grid = generate_multi_grid(_, multi_grid)))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
    def freeze(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    

def ResNet101(pretrained = True):
    resnet101 = DenseFeatureExtractor(BasicBlock, [3,4,23,3])

    if pretrained:
        saved_state_dict = torch.load('./pretrained_models/resnet101-imagenet.pth')
        new_params = resnet101.state_dict().copy()

        for i in saved_state_dict:
            i_parts = i.split('.')
            if not i_parts[0] == 'fc':
                new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

        resnet101.load_state_dict(new_params)
    
    return resnet101





class FullImageEncoder(nn.Module):
    """
    A Full Image Encoder, where we first use an average pooling layer, followed by a fully connected layer,
    then add a conv layer with kernel size of |x|,
    finally, copy the feature vector to F along spatial dimensions
    """
    def __init__(self):
        super(FullImageEncoder, self).__init__()
        self.avgpooling = nn.AvgPool2d(8, stride = 8, padding = (4, 2))
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc = nn.Linear(2048 * 6 * 5, 512)
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(512, 512, 1)
        self.upsample = nn.UpsamplingBilinear2d(size = (33, 45))
                        

        weights_int(self.modules())

    def forward(self, x):
        x1 = self.avgpooling(x)
        x2 = self.dropout(x1)
        x3 = x2.view(-1, 2048 * 6 * 5)
        x4 = self.relu(self.fc(x3))

        x4 = x4.view(-1, 512, 1, 1)
        x5 = self.conv1(x4)

        out = self.upsample(x5)

        return out



class SceneUnderstandingModular(nn.Module):
    """
    Consists of three parallel components:
    1. Atrous Spatial Pyramid Pooling (3 layers, dilations rates are 6, 12, 18, kernel size = 3)
    2. Cross - channel learner (Pure 1x1 convolutional branch)
    3. Full-image encoder

    The obtained features are concatenated, and two additional conv layers with kerner size of 1x1 are added
    """

    def __init__(self):
        super(SceneUnderstandingModular, self).__init__()
        self.encoder = FullImageEncoder()

        self.aspp0 = nn.Sequential(
            nn.Conv2d(2048, 512, 1), # 1.
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 1), # 2.
            nn.ReLU(inplace=True)
        )


        self.aspp1 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding = 6, dilation = 6), # 1.
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 1), # 2.
            nn.ReLU(inplace=True)
        )

        self.aspp2 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding = 12, dilation = 12), # 1.
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 1), # 2.
            nn.ReLU(inplace=True)
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding = 18, dilation = 18), # 1.
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 1), # 2.
            nn.ReLU(inplace=True)
        )

        self.concat_process = nn.Sequential(nn.Dropout2d(p=0.5),
            # Concatenate 1
            nn.Dropout2d(p = 0.5),
            nn.Conv2d(512 * 5, 2048, 1),
            nn.ReLU(inplace = True),
            

            # Concatenate 2
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, 136, 1),
            nn.UpsamplingBilinear2d(size = (257, 353))
        )

        weights_int(self.modules())

        def forward(self, out):
            out1 = self.encoder(out)

            out2 = self.aspp0(out)
            out3 = self.aspp1(out)
            out4 = self.aspp2(out)
            out5 = self.aspp3(out)
            
            out6 = torch.cat((out1, out2, out3, out4, out5), dim = 1)

            out = self.concat_process(out6)

            return out


class OrdinalRegressionLayer(nn.Module):
    def __int__(self):
        super(OrdinalRegressionLayer, self).__init__()
    
    def forward(self, x):
        """
        param: NxHxWxC, N is batch_size, C is channels of features
        return: ord_labels = ordinal outputs for each location, with size NxHxWxC,
                where C = 2K, K is interval of SID
        """

        N, C, H, W = x.size()
        ord_num = C // 2

        A = x[:, ::2, :, :].clone()
        B = x[:, 1::2, :, :].clone()

        A = A.view(N, 1, ord_num * H * W)
        B = B.view(N, 1, ord_num * H * W)

        C = torch.cat((A, B), dim = 1)
        C = torch.clamp(C, min = 1e-8, max = 1e8)

        ord_c = nn.functional.softmax(C, dim = 1)

        ord_c1 = ord_c[:, 1, :].clone()
        ord_c1 = ord_c1.view(-1, ord_num, H, W)
        decode_c = torch.sum((ord_c1 > 0.5), dim = 1).view(-1, 1, H, W)
        return decode_c, ord_c1


class DORN(nn.Module):
    def __init__(self, output_size = (257, 353), channel = 3, pretrained = True, freeze = True):
        super(DORN, self).__init__()

        self.output_size = output_size
        self.channel = channel
        self.feature_extractor = ResNet101(pretrained = pretrained)
        self.SceneUnderstanding = SceneUnderstandingModular()
        self.OrdinalRegression = OrdinalRegressionLayer()

    def forward(self, x):
        # The whole network is consisted of two parts:
        # A Dense Feature Extractor and a Scene Understanding Modular

        x = self.feature_extractor(x)
        x = self.SceneUnderstanding(x)

        depth_labels, ord_labels = self.OrdinalRegression(x)

        return depth_labels, ord_labels

if __name__ == "__main__":
    model = DORN(pretrained=False)
    model = model.cuda()
    model.eval()
    image = torch.randn(1, 3, 257, 353)
    image = image.cuda()
    with torch.no_grad():
        out0, out1 = model(image)
    print('out0 size:', out0.size())
    print('out1 size:', out1.size())

    print(out0)

