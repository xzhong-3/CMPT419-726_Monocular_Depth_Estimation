'''
To evaluate accuracy of estimated depth maps, an additional measure
that is sensitive to positional errors of egdes which will be overlooked
by the above error is proposed.
Sobel operator is applied to both of the estimated and true depth maps,
and the apply a threshold to them to identify pixels by 'pixels on edges'
threshold: 0.25, 0.5, 1
3x3 horizontal and vertical Sobel operator
'''

import torch 
import torch.nn as nn
import numpy as np 

class Sobel(nn.Module):

    def __init__(self):
        super(Sobel, self).__init__()

        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1, stride=1, bias=False)
        edge_x = np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ])
        edge_y = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ])
        edge = np.stack((edge_x, edge_y))
        edge = torch.from_numpy(edge).float().view(2,1,3,3)
        self.conv.weight = nn.Parameter(edge)

        for p in self.parameters():
            p.requires_grad = False


    def forward(self, x):

        out = self.conv(x)
        out = out.contiguous().view(-1,2,x.size(2),x.size(3))

        return out