import os
from dotenv import load_dotenv
import time
import random
import wandb

from Seg3D import Seg3D

from engine import Segmentor
from dataloader import LungDataset_3D_high_res, slab_loader
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from torch import nn
from torch.cuda import amp
import torch
from torchsummary import summary
from sklearn import model_selection

import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)

from tqdm.auto import tqdm

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def init_weights(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)

def wandb_config():
    config = wandb.config
    # ENV
    config.data_path = '/data4/inqlee0704'
    # config.data_path = os.getenv('VIDA_PATH')
    config.in_file = 'ENV18PM_ProjSubjList_cleaned_IN.in'
    config.test_results_dir = "RESULTS"
    config.name = 'Seg3D_n_case128'
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.model = 'Seg3D_high_only'
    config.activation = 'relu'
    config.optimizer = 'adam'
    # config.scheduler = 'CosineAnnealingWarmRestarts'
    config.loss = 'BCE'
    # config.bce_weight = 0.5
    # config.pos_weight = 1

    config.learning_rate = 0.0005
    config.train_bs = 2
    config.valid_bs = 2
    config.aug = False

    config.save = True
    config.debug = False
    if config.debug:
        config.epochs = 1
        config.project = 'debug'
        config.n_case = 16
    else:
        config.epochs = 20
        config.project = 'lung'
        config.n_case = 128
    return config

# def debugIT(data_loader):
#     iters = len(data_loader)
#     pbar = tqdm(enumerate(data_loader), total=iters)
#     for step, batch in pbar:
#         # print(data_loader.ID)
#         img_dim = batch['image'].shape
#         mask_dim = batch['seg'].shape
#         print(f'img shape: {img_dim}')
#         print(f'mask shape: {mask_dim}')

if __name__ == "__main__": 
    load_dotenv()
    seed_everything()
    config = wandb_config()
    wandb.init(project=config.project)
    
    # Data
    df_subjlist = pd.read_csv(os.path.join(config.data_path,config.in_file),sep='\t')
    df_train, df_valid = model_selection.train_test_split(
        df_subjlist[:config.n_case],
        test_size=0.2,
        random_state=42,
        stratify=None)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    train_slabs = slab_loader(df_train)
    valid_slabs = slab_loader(df_valid)
    train_ds = LungDataset_3D_high_res(df_train,train_slabs)
    valid_ds = LungDataset_3D_high_res(df_valid,valid_slabs)
    train_loader = DataLoader(train_ds,
                                batch_size=config.train_bs,
                                shuffle=False,
                                num_workers=0)
    valid_loader = DataLoader(valid_ds,
                                batch_size=config.valid_bs,
                                shuffle=False,
                                num_workers=0)

    # Model #
    # loss 
    # if config.loss == 'BCE':
    #     pos_weight=torch.ones([1])*config.pos_weight
    #     loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(config.device))
    # else:
    #     loss_fn = nn.CrossEntropyLoss()

    model = Seg3D(num_classes=1)
    # model = UNet3D(num_classes=1)
    model.apply(init_weights)
    model.to(config.device)
    summary(model,(1,256,256,32))
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)

    scaler = amp.GradScaler()
    eng = Segmentor(model=model, 
                    optimizer=optimizer,
                    scheduler=None,
                    # loss_fn=loss_fn,
                    device=config.device,
                    scaler=scaler)

    if config.save:
        dirname = f'{config.name}_{time.strftime("%Y%m%d", time.gmtime())}'
        out_dir = os.path.join('RESULTS',dirname)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "lung_Seg3D_low.pth")

    best_loss = np.inf
    # Train
    wandb.watch(eng.model,log='all',log_freq=10)
    for epoch in range(config.epochs):
        trn_loss, trn_dice = eng.train(train_loader)
        val_loss, val_dice = eng.evaluate(valid_loader)
        wandb.log({'epoch': epoch,
         'trn_loss': trn_loss, 'trn_dice': trn_dice,
         'val_loss': val_loss, 'val_dice': val_dice})
        # trn_loss, trn_dice_loss, trn_bce_loss = eng.train(train_loader)
        # val_loss, val_dice_loss, val_bce_loss = eng.evaluate(valid_loader)
        # wandb.log({'epoch': epoch,
        #  'trn_loss': trn_loss, 'trn_dice_loss': trn_dice_loss, 'trn_bce_loss': trn_bce_loss,
        #  'val_loss': val_loss, 'val_dice_loss': val_dice_loss, 'val_bce_loss': val_bce_loss})

        # if config.scheduler == 'ReduceLROnPlateau':
        #     scheduler.step(val_loss)
        eng.epoch += 1
        print(f'Epoch: {epoch}, train loss: {trn_loss:5f}, valid loss: {val_loss:5f}')
        if val_loss < best_loss:
            best_loss = val_loss
            print(f'Best loss: {best_loss} at Epoch: {eng.epoch}')
            if config.save:
                torch.save(model.state_dict(), path)
                wandb.save(path)
