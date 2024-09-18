import torch
import torch.nn as nn
from torch.optim import Adam

from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

import torchvision
from torchvision import models

import pytorch_lightning as pl
from pytorch_lightning  import LightningModule

import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pdb
import time

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()
    '''
    this function is designed for host machine, unable to work within docker because docker dosent provide a GUI
    '''

class ADJSCC(LightningModule):
    def __init__(self,encoder,decoder,loss_module_G,channel,lr_scheduler_type,lr_G):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_module_G = loss_module_G
        self.channel = channel#simulate channel
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_G = lr_G

    def forward(self,image,snr):
        encoded = self.encoder.forward(image,snr)
        received = self.channel.forward(encoded,snr)
        decoded = self.decoder.forward(received,snr)
        return decoded
        
    def training_step(self,batch):
        snr = random.randint(0, 20)
        image,_ = batch
        decoded = self.forward(image=image,snr=snr)
        loss_G = self.loss_module_G(image,decoded)
        loss_G = loss_G.to(self.device)
        self.log('training loss G',loss_G,on_step = False, on_epoch = True,prog_bar = True,logger = True, sync_dist=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning rate',current_lr,on_step = False, on_epoch = True,prog_bar = True,logger = True, sync_dist=True)
        return loss_G

    def validation_step(self,batch,batch_idx):
        snr = int(15)
        source_image,_ = batch
        decoded = self.forward(source_image,snr)
        loss_G = self.loss_module_G(source_image,decoded)
        loss_G = loss_G.to(self.device)
        mse = torch.mean(((source_image/2+0.5) - (decoded/2+0.5)) ** 2, dim=[1, 2, 3])
        psnr = 10*torch.log10(1/mse)
        self.logger.experiment.add_image('source',source_image[0]/2+0.5,self.current_epoch)
        self.logger.experiment.add_image('val_res',decoded[0]/2+0.5,self.current_epoch)
        self.log('psnr',torch.mean(psnr),on_step = False, on_epoch = True,prog_bar = True,logger = True, sync_dist=True)
        self.log('val_loss_G',loss_G,on_step = False, on_epoch = True,prog_bar = True,logger = True, sync_dist=True)
        return loss_G

    def configure_optimizers(self):
        optimizer_G = Adam(self.parameters(), lr=self.lr_G)
        lr_scheduler_type = self.lr_scheduler_type
        if 'step' in lr_scheduler_type.lower():
            scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G,step_size = 400,gamma = 0.1)
        else:
            pass
        optim_dict = {'optimizer':optimizer_G,'lr_scheduler':scheduler_G}
        return optim_dict