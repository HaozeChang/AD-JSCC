import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model.module.gdn import GDN
from model.module.SEblock import AFlayer


import pdb
import math

torch.autograd.set_detect_anomaly(True)

    
class DecodingBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,acti = 'PR'):
        super(DecodingBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,padding)
        self.igdn = GDN(ch=out_channels,inverse=True)
        self.acti = acti
        if acti=='none':
            self.activation = None    
        elif acti=='sig':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.igdn(x)
        if self.acti=='none':
            pass
        elif self.acti=='sig':
            x = self.activation(x)
        else:
            x = self.activation(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self,in_channels):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.channels = [256,256,256,256,3]
        self.block1 = DecodingBlock(in_channels=self.in_channels,out_channels=self.channels[0],kernel_size=5,stride=1,padding =2,acti='PR')
        self.block2 = DecodingBlock(in_channels=self.channels[0],out_channels=self.channels[1],kernel_size=5,stride=1,padding =2,acti='PR')
        self.block3 = DecodingBlock(in_channels=self.channels[1],out_channels=self.channels[2],kernel_size=5,stride=1,padding =2,acti='PR')
        self.block4 = DecodingBlock(in_channels=self.channels[2],out_channels=self.channels[3],kernel_size=5,stride=2,padding =2,acti='PR')
        self.block5 = DecodingBlock(in_channels=self.channels[3],out_channels=self.channels[4],kernel_size=8,stride=2,padding =2,acti='sig')# kernel_size = 8?
        
        self.se1 = AFlayer(channel=self.channels[0],reduction=16)
        self.se2 = AFlayer(channel=self.channels[1],reduction=16)
        self.se3 = AFlayer(channel=self.channels[2],reduction=16)
        self.se4 = AFlayer(channel=self.channels[3],reduction=16)
        # original channel: 400 32 32 32 16
        self.blocks = [self.block1,self.block2,self.block3,self.block4,self.block5]
        self.ses = [self.se1,self.se2,self.se3,self.se4]

    def forward(self,x,snr):
        x = x.unsqueeze(2).unsqueeze(3) #reshape to b,c,1,1
        b,c,_,_ = x.shape
        h = int(math.sqrt(c/self.in_channels))
        w = h
        c = self.in_channels
        x = x.view(b,c,h,w)
        for i in range(4):
            x = self.blocks[i](x)
            x = self.ses[i](x,snr)
        x = self.block5(x)
        return x
