import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model.module.SEblock import AFlayer
from model.module.gdn import GDN

import pdb
import math

#torch.autograd.set_detect_anomaly(True) 
def power_normalize(x):
    x = torch.mean(x, (-2, -1))
    b, c = x.shape  # Assuming x has shape [batch_size, 20]
    alpha = math.sqrt(c)
    energy = torch.norm(x, p=2, dim=1)# Calculate the L2 norm of each one-dimensional vector
    alpha = alpha / energy.unsqueeze(1)# Calculate the normalization factor alpha for each vector
    x_normalized = alpha * x# Apply alpha to each vector
    return x_normalized

class EncodingBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,acti = 'PR'):
        super(EncodingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding = padding)
        self.gdn = GDN(ch=out_channels)
        self.acti = acti
        if acti=='none':
            self.activation = None    
        elif acti=='sig':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.gdn(x)
        if self.acti=='none':
            pass
        elif self.acti=='sig':
            x = self.activation(x)
        else:
            x = self.activation(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, out_channels):
        super(Encoder, self).__init__()
        self.out_channels = out_channels
        self.channels = [3,256,256,256,256]
        self.block1 = EncodingBlock(in_channels=self.channels[0],out_channels=self.channels[1],kernel_size=9,stride=2,padding = 4,acti='PR')
        self.block2 = EncodingBlock(in_channels=self.channels[1],out_channels=self.channels[2],kernel_size=5,stride=2,padding = 2,acti='PR')
        self.block3 = EncodingBlock(in_channels=self.channels[2],out_channels=self.channels[3],kernel_size=5,stride=1,padding = 2,acti='PR')
        self.block4 = EncodingBlock(in_channels=self.channels[3],out_channels=self.channels[4],kernel_size=5,stride=1,padding = 2,acti='PR')
        self.block5 = EncodingBlock(in_channels=self.channels[4],out_channels=self.out_channels,kernel_size=5,stride=1,padding = 2,acti='none')
        self.se1 = AFlayer(channel=self.channels[1],reduction=16)
        self.se2 = AFlayer(channel=self.channels[2],reduction=16)
        self.se3 = AFlayer(channel=self.channels[3],reduction=16)
        self.se4 = AFlayer(channel=self.channels[4],reduction=16)
        # original channels:3 16 32 32 32 400
        self.blocks = [self.block1,self.block2,self.block3,self.block4,self.block5]
        self.ses = [self.se1,self.se2,self.se3,self.se4]

    def forward(self,x,snr):
        for i in range(4):
            x = self.blocks[i](x)
            x = self.ses[i](x,snr)
        x = self.block5(x)
        b,c,h,w=x.shape
        x = x.view(b,c*h*w,1,1)
        x = power_normalize(x)
        return x        