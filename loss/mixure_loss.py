import torch
import torch.nn as nn
import torch.nn.functools as F
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

import numpy as np

import pdb

#----------------------MSE loss--------------------
class MSEImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn1 = torch.nn.MSELoss(reduction='mean')

    def forward(self,target_image,received_image):
        loss1 = self.loss_fn1(target_image.float(), received_image.float())
        return loss1
#----------------------SSIM loss--------------------
class SSIMLossImage(nn.Module):
    def __init__(self):
        super().__init__()    
    
    def forward(self,target_image,received_image):
        ssimer = SSIM(data_range=1.0, reduction='none').to(target_image.device)  # ssim for each image
        ssim = ssimer(target_image, received_image).clone()
        ssim = ssim.mean()
        return (1-ssim)

#----------------------mixed MSE and SSIM loss--------------------
class MixtureLossImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn1 = torch.nn.MSELoss(reduction='mean')
        self.a = nn.Parameter(torch.tensor(0.5))
    
    def forward(self,target_image,received_image):
        loss1 = self.loss_fn1(target_image.float(), received_image.float())
        ssimer = SSIM(data_range=1.0, reduction='none').to(target_image.device)
        ssim = ssimer(target_image, received_image).clone()
        ssim = ssim.mean()
        a_clamped = torch.clamp(self.a, 0, 1)
        return a_clamped * loss1 + (1 - a_clamped) * (1 - ssim)  #optimize 1-SSIM rather than SSIM
        
    
#---------------------------------CE loss -------------------------
class CELossAck(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,ack,similarity):
        loss = self.loss_fn(ack, similarity)
        return loss
    
#---------------------------------binary CE loss -------------------------
class BCELossAck(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self,ack,similarity):
        """
        need [batch,2] size as input and target to calculate. 2 as one-hot, input as two logits representing the score
        """
        loss = self.loss_fn(ack.float(), similarity.float())
        return loss
