# train_util.py
# ##############################################################################
# 20210307, In Kyu Lee
# Desc: Deep Learning train utilities
# ##############################################################################
# Optimizer:
#  - Adam
#  - AdamP
# Scheduler:
#  - ReduceLROnPlateau
#  - CosineAnnealingLR
#  CosineAnnealingWarmRestarts
# ##############################################################################
# How to use:
# from train_util import *
# optimizer = get_optimizer(model,CFG)
# scheduler = get_scheduler(optimizer, CFG)
# ##############################################################################

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR, ReduceLROnPlateau
from adamp import AdamP


# Optimizer
def get_optimizer(model,CFG):
    if CFG.optimizer =='adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=CFG.lr)
    elif CFG.optimizer == 'adamp':
        optimizer = AdamP(model.parameters(),lr=CFG.lr)
    return optimizer

# Scheduler
def get_scheduler(optimizer,CFG):
    if CFG.scheduler=='ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=CFG.patience)

    elif CFG.scheduler=='CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, 
                                        T_max=CFG.T_max,
                                        eta_min=CFG.min_lr,
                                        last_epoch=-1)

    elif CFG.scheduler=='CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                T_0=CFG.T_0, 
                                                T_mult=1, 
                                                eta_min=CFG.min_lr,
                                                last_epoch=-1)
    return scheduler