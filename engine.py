from tqdm.auto import tqdm
import torch
from torch.cuda import amp
from sklearn import metrics
import numpy as np

from torch import nn

def Dice3d(a,b):
    # print(f'pred shape: {a.shape}')
    # print(f'target shape: {b.shape}')
    intersection =  np.sum((a!=0)*(b!=0))
    volume = np.sum(a!=0) + np.sum(b!=0)
    if volume == 0:
        return -1
    return 2.*float(intersection)/float(volume)

def cal_dice(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return dice.mean()

def cal_dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def cal_loss(outputs, targets, bce_weight=0.5):
    BCE_fn = nn.BCEWithLogitsLoss()
    bce_loss = BCE_fn(outputs, targets)
    preds = torch.sigmoid(outputs)
    dice = cal_dice(preds, targets)
    return bce_loss, dice
    
    # loss = bce_loss * bce_weight + dice_loss * (1-bce_weight)

    # return loss, bce_loss, dice_loss

class Segmentor:
    def __init__(self,model,optimizer=None,scheduler=None,device=None,scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # self.loss_fn = loss_fn
        self.device = device
        self.scaler = scaler
        self.epoch = 0

    def train(self, data_loader):
        self.model.train()
        epoch_loss = 0
        epoch_dice = 0
        # epoch_bce_loss = 0
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=iters)
        for step, batch in pbar:
            self.optimizer.zero_grad()
            inputs = batch['image'].to(self.device,dtype=torch.float)
            # if BCEwithLogitsLoss,
            targets = batch['seg'].to(self.device, dtype=torch.float)
            # if CrossEntropyLoss,
            # targets = batch['seg'].to(self.device)
            with amp.autocast():
                outputs = self.model(inputs)
                # loss, bce_loss, dice_loss = cal_loss(outputs, targets)
                loss, dice = cal_loss(outputs, targets)

            # outputs = torch.sigmoid(outputs)
            # preds = np.round(outputs.cpu().detach().numpy())
            # preds = np.squeeze(preds, axis=1)
            # targets = targets.cpu().detach().numpy()
            # targets = np.squeeze(targets,axis=1)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # self.scheduler.step(self.epoch+step/iters)
            epoch_loss += loss.item()
            epoch_dice += dice.item()
            # epoch_bce_loss += bce_loss.item()
            pbar.set_description(f'loss:{loss:.3f}, dice:{dice:.3f}') 
            # pbar.set_description(f'loss:{loss:.3f}, dice loss:{dice_loss:.3f}, bce loss:{bce_loss:.3f}') 
        # return epoch_loss/iters, epoch_dice_loss/iters, epoch_bce_loss/iters
        return epoch_loss/iters, epoch_dice/iters

    def evaluate(self, data_loader):
        self.model.eval()
        epoch_loss = 0
        epoch_dice = 0
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader),total=iters)
        with torch.no_grad():
            for step, batch in pbar:
                inputs = batch['image'].to(self.device,dtype=torch.float)
                # if BCEwithLogitsLoss,
                targets = batch['seg'].to(self.device, dtype=torch.float)
                # if CrossEntropyLoss,
                # targets = batch['seg'].to(self.device)
                outputs = self.model(inputs)
                # loss = self.loss_fn(outputs, targets)
                loss, dice = cal_loss(outputs, targets)
                epoch_loss += loss.item()
                epoch_dice += dice.item()
                # preds = np.argmax(outputs.cpu().detach().numpy(),axis=1)
                # outputs = torch.sigmoid(outputs)
                # preds = np.round(outputs.cpu().detach().numpy())
                # preds = np.squeeze(preds, axis=1)
                # targets = targets.cpu().detach().numpy()
                # targets = np.squeeze(targets,axis=1)
                # dice = Dice3d(preds,targets)
                pbar.set_description(f'loss:{loss:.3f}, dice:{dice:.3f}') 
                # pbar.set_description(f'loss:{loss:.3f}, dice loss:{dice_loss:.3f}, bce loss:{bce_loss:.3f}') 
            return epoch_loss/iters, epoch_dice/iters

    def predict(self, data_loader):
        self.model.eval()
        batch_preds = []
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader),total=iters)
        with torch.no_grad():
            for step, batch in pbar:
                inputs = batch['image'].to(self.device,dtype=torch.float)
                # if BCEwithLogitsLoss,
                # if CrossEntropyLoss,
                # targets = batch['seg'].to(self.device)
                outputs = self.model(inputs)
                preds = torch.sigmoid(outputs)
                preds = preds.cpu().detach().numpy()
                preds = np.squeeze(preds, axis=1)
                batch_preds.append(preds)
                return batch_preds
