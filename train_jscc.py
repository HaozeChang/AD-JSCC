import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from model.ADjscc import ADJSCC
from model.module.encoder import Encoder
from model.module.decoder import Decoder
from channels.AWGN import AWGNChannel
from channels.Rayl import RayleighChannel
from loss.mixure_loss import MSEImageLoss

from argparse import ArgumentParser
from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np
import pdb

def boolean_transfer(string): #sometimes often comes up with string / bool dosen't match, a pain in the ass!
    if type(string)==str:
        if 'false' in string.lower():
            return False
        else:
            return True
    
def list_transfer(target_list):# transfer ele into an list
    res = []
    for ele in target_list:
        if ele !=',' and ele !='['and ele !=']' and ele!=' ':
            res.append(int(ele))
    return res

def device_transfer(device):# change device, because pl allows you to asign specific device such as GPU num [0,2]
    """
    transfer device data type
    """
    if '[' in device:
        return list_transfer(device)
    else:
        return device

def update_model(optimizer,new_lr,milestones):# update model lr
    """
    update model optimizers if needed(lr,lr_scheduler)
    """
    if isinstance(optimizer, list):
        # if multiple optimizers
        for opt in optimizer:
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr
    elif isinstance(optimizer, dict):
        # if return type is a dict with optimizer and lr_scheduler
        opt = optimizer['optimizer']
        for param_group in opt.param_groups:
            param_group['lr'] = new_lr
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,milestones = list_transfer(milestones),gamma = 0.1)
        optimizer['lr_scheduler'] = lr_scheduler
    else:
        # if only one optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return optimizer

def imshow(img):
    """
    denormalize and show image
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train(hparams):
    """
    input : hparams
    output: training
    """
    # define the basic parts
    encoder = Encoder(out_channels=8)
    decoder = Decoder(in_channels=8)
    # determine whether to use the ckpt
    if 'awgn' in hparams.channel.lower():
        channel = AWGNChannel()
        print("awgn")
    else:
        channel = RayleighChannel()
        print("rayl")
    ckpt_addr = hparams.ckpt_addr
    print(colored('====>','red')+ckpt_addr)
    if len(ckpt_addr)==0:
        model = ADJSCC(
                    encoder=encoder,decoder=decoder,
                    loss_module_G=MSEImageLoss(),
                    channel=channel,
                    lr_scheduler_type = 'step',
                    lr_G = 1e-4
                )
        
    else:
        model = ADJSCC.load_from_checkpoint(ckpt_addr,
                    encoder=encoder,decoder=decoder,
                    loss_module_G=MSEImageLoss(),
                    channel=channel,
                    lr_scheduler_type = 'step',
                    lr_G = 1e-4
                )
    print(colored('========》','red')+"model done")
    # print model summary
    if boolean_transfer(hparams.print_model_summary):
        print(ModelSummary(model,-1))

    # define model hparams
    batch_size = int(hparams.batch_size)

    # build up Dataloader instance
    # TODO: if needed, initialize dataloader
    '''
    dataloader = Dataloader(batch_size)
    dataloader.setup(root='/home/COCO/images/train2014/')
    train_loader = dataloader.train_dataloader()
    val_loader = dataloader.val_dataloader()
    '''
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./cifar_data', train=True,download=True, transform=transform)
    valset = torchvision.datasets.CIFAR10(root='./cifar_data', train=False,download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=15,pin_memory = False)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,shuffle=False, num_workers=15,pin_memory = False)
    # create logger instance
    if 'awgn' in hparams.channel.lower():
        logger = TensorBoardLogger("logs",name = "JSCC/AWGN")
    else:
        logger = TensorBoardLogger("logs",name = "JSCC/RAYL")
    # define call backs
    callbacks = [pl.callbacks.ModelCheckpoint(every_n_epochs=int(hparams.save_ckpt_every_n_epochs),save_top_k = 2,monitor="val_loss_G",mode="min"),
                EarlyStopping(monitor = 'val_loss_G',min_delta=0.0005, patience=50)]
    #define profiler:
    profiler = hparams.profiler
    #define trainer
    if boolean_transfer(hparams.fast_dev_run) == True or hparams.fast_dev_run.isdigit():
        # in this case, we run some simply test to make sure no crash
        '''
        Here the logic is a littel chao, but works well in both fast_dev_run = True \ False situation
        
        '''
        if hparams.fast_dev_run.isdigit():
            fast_dev_run = int(hparams.fast_dev_run)
        else:
            fast_dev_run = boolean_transfer(hparams.fast_dev_run)
        trainer = Trainer(max_epochs = hparams.max_epoches,
                        devices=device_transfer(hparams.device), accelerator="gpu",
                        default_root_dir = 'snapshots/',
                        enable_checkpointing=True,
                        enable_progress_bar=True, 
                        enable_model_summary = True,
                        callbacks = callbacks,
                        logger = logger,
                        check_val_every_n_epoch=int(hparams.check_val_every_n_epoch),
                        fast_dev_run= fast_dev_run,
                        profiler = profiler,
                        #strategy='ddp_find_unused_parameters_true'
                        strategy = DDPStrategy(find_unused_parameters=True)
                        )
    else:
        trainer = Trainer(max_epochs = hparams.max_epoches,
                        devices=device_transfer(hparams.device), accelerator="gpu",
                        default_root_dir = 'snapshots/',
                        enable_checkpointing=True,
                        enable_progress_bar=True, 
                        enable_model_summary = True,
                        callbacks = callbacks,
                        logger = logger,
                        check_val_every_n_epoch=int(hparams.check_val_every_n_epoch),
                        profiler = profiler,
                        #strategy='ddp_find_unused_parameters_true'
                        strategy = DDPStrategy(find_unused_parameters=True)
                        )
    # start training!
    print(colored('=======================','yellow')+"Starting training!"+colored('=======================','yellow'))
    trainer.fit(model,train_loader, val_loader)

def main(hparams):
    train(hparams)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt_addr", type = str,default='')
    parser.add_argument("--batch_size",type = int, default=16)
    parser.add_argument("--device",type = str, default=[1])
    parser.add_argument("--max_epoches", type = int, default=10)
    parser.add_argument("--plot_lr", type = str ,default=False)
    parser.add_argument("--check_val_every_n_epoch", type = int,default=10)
    parser.add_argument("--save_ckpt_every_n_epochs", type = int,default=10)
    parser.add_argument("--milestones", type=list,default=None)
    parser.add_argument("--channel", type = str , default=None)
    parser.add_argument("--fast_dev_run", type = str , default=False)
    parser.add_argument("--profiler",type=str,default=None)
    parser.add_argument("--lr_scheduler_type",type=str,default=None)
    parser.add_argument("--print_model_summary",type=str,default=False)
    parser.add_argument("--find_init_lr",type=str,default=False)
    args = parser.parse_args()
    main(args)
