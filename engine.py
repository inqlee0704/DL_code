# engine.py
# ##############################################################################
# 20210307, In Kyu Lee
# Desc: Deep Learning Engines
# ##############################################################################
# Engines:
#  - Classifier: Classification
#  - Segmentor: Semantic Segmentation
#  - Detector: Object Detection (Need to be updated)
# ##############################################################################
# How to use:
# engine = Classifier(model,optimizer,scheduler,loss_fn,device,scaler)
# for epoch in range(epochs):
#   engine.train(data_loader)
#   engine.evaluatate(data_loader)
# ##############################################################################

from tqdm.auto import tqdm
import torch
from torch.cuda import amp
from sklearn import metrics
from metrics_util import Dice3d
import numpy as np

class Classifier:
    def __init__(self,model,optimizer,scheduler,loss_fn,device,scaler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.scaler = scaler
        self.epoch = 0
        
    def train(self, data_loader):
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=iters)
        for step, batch in pbar:
            self.optimizer.zero_grad()
            inputs = batch['img'].to(self.device,dtype=torch.float)
            targets = batch['targets'].to(self.device)
            with amp.autocast():
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
            preds = torch.argmax(outputs,dim=1).cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
            acc = metrics.accuracy_score(targets,preds)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step(self.epoch+step/iters)
            epoch_loss += loss.item()
            epoch_acc += acc
            pbar.set_description(f'loss:{loss:.2f}, acc:{acc:.2f}')         
        return epoch_loss/iters, epoch_acc/iters

    def evaluate(self, data_loader):
        self.model.eval()
        final_loss = 0
        final_acc = 0
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=iters)
        with torch.no_grad():
            for step, batch in pbar:
                inputs = batch['img'].to(self.device,dtype=torch.float)
                targets = batch['targets'].to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                preds = torch.argmax(outputs,dim=1).cpu().detach().numpy()
                targets = targets.cpu().detach().numpy()
                acc = metrics.accuracy_score(targets,preds)
                final_loss += loss.item()
                final_acc += acc
                pbar.set_description(f'loss:{loss:.2f}, acc:{acc:.2f}')  
        return final_loss/iters, final_acc/iters

class Segmentor:
    def __init__(self,model,optimizer,scheduler,loss_fn,device,scaler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.scaler = scaler
        self.epoch = 0

    def train(self, data_loader):
        self.model.train()
        epoch_loss = 0
        epoch_dice = 0
        skip = 0
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=iters)
        for step, batch in pbar:
            self.optimizer.zero_grad()
            inputs = batch['image'].to(self.device,dtype=torch.float)
            targets = batch['seg'].to(self.device)
            with amp.autocast():
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets[:, 0, :, :])
            preds = np.argmax(outputs.cpu().detach().numpy(),axis=1)
            targets = targets.cpu().detach().numpy()
            targets = np.squeeze(targets,axis=1)
            dice = Dice3d(preds,targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step(self.epoch+step/iters)
            epoch_loss += loss.item()
            if dice==-1:
                skip += 1
            else:
                epoch_dice += dice
            pbar.set_description(f'loss:{loss:.2f}, dice:{dice:.4f}') 
        return epoch_loss/iters, epoch_dice/(iters-skip)

    def evaluate(self, data_loader):
        self.model.eval()
        epoch_loss = 0
        epoch_dice = 0
        skip = 0
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader),total=iters)
        with torch.no_grad():
            for step, batch in pbar:
                inputs = batch['image'].to(self.device,dtype=torch.float)
                outputs = self.model(inputs)
                targets = batch['seg'].to(self.device)
                loss = self.loss_fn(outputs, targets[:, 0, :, :])
                epoch_loss += loss.item()
                preds = np.argmax(outputs.cpu().detach().numpy(),axis=1)
                targets = targets.cpu().detach().numpy()
                targets = np.squeeze(targets,axis=1)
                dice = Dice3d(preds,targets)
                if dice==-1:
                    skip += 1
                else:
                    epoch_dice += dice
                pbar.set_description(f'loss:{loss:.2f}, dice:{dice:.4f}') 
            return epoch_loss/iters, epoch_dice/(iters-skip)
 

class Detector:
    def __init__(self,model,optimizer,scheduler,loss_fn,device,scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss_fn = loss_fn
        self.scaler = scaler
        self.epoch = 0
    
    def train(self,data_loader):
        return None
    
    def evaluate(self,data_loader):
        return None