'''
Data augmentation method: 
- flip: the RGB and the depth image are both horizontally flipped with 0.5 probability
- rotation: the RGB and the depth image are both rotated by a random degree r∈[-5,5]
- color jitter: brightness, contrast, and saturation values of the RGB image are randomly scaled by c∈[0.6,1.4]

downsample images from 640x480 to 320x240 bilinear interpolation
crop central parts to 304x228

training - 249, testing - 215

'''

import torch 
import torch.nn as nn
import random
import numpy as np 
import pandas as pd 
from PIL import Image, ImageOps
import scipy.ndimage as ndimage
from torchvision import transforms


__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}


class HorizontalFlip(object):

    def __call__(self, samples):

        image, depth = samples['image'], samples['depth']

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}


class Rotate(object):

    def __init__(self, angle, order=2, reshape=False):
        self.angle = angle
        self.order = order
        self.reshape = reshape


    def __call__(self, samples):

        image, depth = samples['image'], samples['depth']

        angle1 = random.uniform(-self.angle, self.angle)
        ## angle2 = angle1 * np.pi / 180

        image = ndimage.interpolation.rotate(image, angle1, reshape=self.reshape, order=self.order)
        depth = ndimage.interpolation.rotate(depth, angle1, reshape=self.reshape, order=self.order)

        image = Image.fromarray(image)
        depth = Image.fromarray(depth)

        return {'image': image, 'depth': depth}


class Brightness(object):

    def __init__(self, brightness):
        self.brightness = brightness


    def __call__(self, image):

        grayscale = image.new().resize_as_(image).zero_()
        a = random.uniform(-self.brightness, self.brightness)

        return image.lerp(grayscale, a)



class Grayscale(object):

    def __call__(self, image):

        grayscale = image.clone()
        grayscale[0].mul_(0.299).add_(0.587, grayscale[1]).add_(0.144, grayscale[2])
        grayscale[1].copy_(grayscale[0])
        grayscale[2].copy_(grayscale[0])

        return grayscale


class Contrast(object):

    def __init__(self, contrast):
        self.contrast = contrast


    def __call__(self, image):

        grayscale = Grayscale()(image)
        grayscale.fill_(grayscale.mean())
        a = random.uniform(-self.contrast, self.contrast)

        return image.lerp(grayscale, a)




class Saturation(object):

    def __init__(self, saturation):
        self.saturation = saturation


    def __call__(self, image):

        grayscale = Grayscale()(image)
        a = random.uniform(-self.saturation, self.saturation)

        return image.lerp(grayscale, a)



class RandomOrder(object):

    def __init__(self, transforms):
        self.transforms = transforms

    
    def __call__(self, samples):

        image, depth = samples['image'], samples['depth']

        if self.transforms is None:
            return {'image': image, 'depth': depth}

        order = torch.randperm(len(self.transforms))
        for i in order:
            image = self.transforms[i](image)

        return {'image': image, 'depth': depth}


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []

        if brightness is not 0:
            self.transforms.append(Brightness(brightness))

        if contrast is not 0:
            self.transforms.append(Contrast(contrast))

        if saturation is not 0:
            self.transforms.append(Saturation(saturation))



class Rescale(object):

    def __init__(self, size):
        self.size = size


    def __call__(self, samples):

        image, depth = samples['image'], samples['depth']

        image = self.rescale(image, self.size, Image.BILINEAR)
        depth = self.rescale(depth, self.size, Image.NEAREST)

        return {'image': image, 'depth': depth}


    def rescale(self, image, size, interpolation=Image.BILINEAR):

        width, height = image.size
        if (width <= height and width == size) or (height <= width and height == size):
            return image

        if width < height:
            new_width = size
            new_height = int(size * height / width)
            return image.resize((new_width, new_height), interpolation)

        else:
            new_height = size
            new_width = int(size * width / height)
            return image.resize((new_width, new_height), interpolation)



class CenterCrop(object):

    def __init__(self, image_size, depth_size):
        super().__init__()

        self.image_size = image_size
        self.depth_size = depth_size


    def __call__(self, samples):

        image, depth = samples['image'], samples['depth']

        image = self.center_crop(image, self.image_size)
        depth = self.center_crop(depth, self.image_size)

        w, h = self.depth_size
        depth = depth.resize((w, h))

        return {'image': image, 'depth': depth}



    def center_crop(self, image, size):

        width, height = image.size
        new_width, new_height = size

        if width == new_width and height == new_height:
            return image

        x1 = int(round((width - new_width) / 2.))
        y1 = int(round((height - new_height) / 2.))

        image = image.crop((x1, y1, new_width+x1, new_height+y1))

        return image



class ImageToTensor(object):

    def __init__(self, is_test=False):
        self.is_test = is_test


    def __call__(self, samples):

        image, depth = samples['image'], samples['depth']

        image = self.to_tensor(image)

        if self.is_test:
            depth = self.to_tensor(depth).float()/1000
        else:
            depth = self.to_tensor(depth).float()*10

        return {'image': image, 'depth': depth}


    def to_tensor(self, picture):

        if isinstance(picture, np.ndarray):
            image = torch.from_numpy(picture.transpose((2, 0, 1)))
            #print("!!!!!!!!!!ndarrray")
            return image.float().div(255)

        if picture.mode == 'I':
            image = torch.from_numpy(np.array(picture, np.int32, copy=False))
            #print("!!!!!!!!!!IIIIIII")
        elif picture.mode == 'I;16':
            image = torch.from_numpy(np.array(picture, np.int16, copy=False))
            #print("!!!!!!!!!!16161616616")
        else:
            ##print("!!!!!!!!!!otherotherother")
            image = torch.ByteTensor(
                torch.ByteStorage.from_buffer(picture.tobytes())
            )

        if picture.mode == 'YCbCr':
            #print("!!!!!!!!!!YCYCYCYCYCYYC")
            n_channels = 3
        elif picture.mode == 'I;16':
            n_channels = 1
        else:
            n_channels = len(picture.mode)

        image = image.view(picture.size[1], picture.size[0], n_channels)
        image = image.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(image, torch.ByteTensor):
            return image.float().div(255)
        else:
            return image



class Lighting(object):

    def __init__(self, a, eigval, eigvec):
        self.a = a
        self.eigval = eigval
        self.eigvec = eigvec


    def __call__(self, samples):

        image, depth = samples['image'], samples['depth']

        if self.a == 0:
            return image

        alpha = image.new().resize_(3).normal_(0, self.a)
        RGB = self.eigvec.type_as(image).clone().mul(alpha.view(1, 3).expand(3, 3)).mul(self.eigval.view(1, 3).expand(3, 3)).sum(1).squeeze()

        image = image.add(RGB.view(3, 1, 1).expand_as(image))

        return {'image': image, 'depth': depth}



class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    
    def __call__(self, samples):

        image, depth = samples['image'], samples['depth']

        image = self.normalize(image, self.mean, self.std)

        return {'image': image, 'depth': depth}


    def normalize(self, tensor, mean, std):

        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)

        return tensor


class NYUDataset(torch.utils.data.Dataset):

    def __init__(self, file_url, transforms=None):
        self.dataset = pd.read_csv(file_url, header=None)
        self.transforms = transforms
        self.len = len(self.dataset)


    def __getitem__(self, index):

        image_name = self.dataset.ix[index, 0]
        depth_name = self.dataset.ix[index, 1]

        image = Image.open(image_name)
        depth = Image.open(depth_name)

        samples = {'image': image, 'depth': depth}

        if self.transforms is not None:
            samples = self.transforms(samples)

        return samples


    def __len__(self):

        return self.len



def training_data(file_url, batch_size=8):

    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    training_set = NYUDataset(file_url=file_url,
                              transforms=transforms.Compose([
                                  Rescale(240),
                                  HorizontalFlip(),
                                  Rotate(5),
                                  CenterCrop([304, 228], [152, 114]), 
                                  ImageToTensor(),
                                  Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
                                  ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                  Normalize(__imagenet_stats['mean'], __imagenet_stats['std'])
                              ]))

    training_loader = torch.utils.data.DataLoader(training_set, batch_size, shuffle=True, num_workers=4, pin_memory=False)

    return training_loader



def testing_data(file_url, batch_size=8):

    testing_set = NYUDataset(file_url=file_url,
                             transforms=transforms.Compose([
                                 Rescale(240), 
                                 CenterCrop([304, 228], [304, 228]),
                                 ImageToTensor(is_test=True),
                                 Normalize(__imagenet_stats['mean'], __imagenet_stats['std'])
                             ]))

    testing_loader = torch.utils.data.DataLoader(testing_set, batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return testing_loader





