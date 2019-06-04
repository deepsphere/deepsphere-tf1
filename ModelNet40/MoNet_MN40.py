#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg

class Model(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        
        self.features = [6, 16, 32, 64, 128, 256, n_classes]
        Nside = 32
        self.nside = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//32, Nside//32]
        self.K = [4]*6
        sequ = []
        for i in range(len(self.features)-2):
            sequ.append(tg.GMMconv(self.features[i], self.features[i+1]), 2, self.K[i], aggr='mean')
            sequ.append(nn.BatchNorm3d(self.features[i+1], affine=True))
            sequ.append(nn.ReLU())
#         self.conv.append(tg.GMMconv(self.features[-3], self.features[-2]), 1, self.K[i], aggr='mean', bias=False)
#         self.bn = nn.BatchNorm3d
#         self.relu = nn.ReLU(inplace=True)
        self.sequential = nn.Sequential(*sequ)
        self.out = nn.Linear(self.features[-2], self.features[-1])  # M
    
    def forward(self, x):
        x = self.sequential(x)
        x = nn.AvgPool2d(x, x.size()[2:])
        x = self.out(x)
        
        return F.log_soft_max(x, dim=1)

def main():
    in_c, out_c = 5, 5
    dim = 3
    kernel_size = 1
    model = Model(40)
    pass

if __name__=='__main__':
    print('hello')
    
    main