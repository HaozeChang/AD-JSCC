import torch
import torch.nn as nn
import pdb

class AFlayer(nn.Module):
    def __init__(self,channel,reduction = 16):
        super(AFlayer,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel+1,channel//reduction,bias = False),
            nn.ReLU(inplace =  True),
            nn.Linear(channel//reduction,channel,bias = False),
            nn.Sigmoid()
        ) 

    def forward(self,x,snr):
        b,c,_,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        snr = torch.full((b, 1), snr)
        snr = snr.to(y.device)
        y = torch.cat([snr,y],1)
        y = self.fc(y)
        y= y.view(b,c,1,1)
        y = x*y.expand_as(x)
        return y