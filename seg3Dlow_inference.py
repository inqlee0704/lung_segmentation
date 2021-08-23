import os
from dotenv import load_dotenv
import time
import random
import wandb

from RecursiveUNet3D import UNet3D
from Seg3D import Seg3D
from medpy.io import load

from engine import Segmentor
from dataloader import LungDataset_3D_low_res
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

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def wandb_config():
    config = wandb.config
    # ENV
    config.data_path = '/data4/inqlee0704'
    # config.data_path = os.getenv('VIDA_PATH')
    config.in_file = 'ENV18PM_ProjSubjList_cleaned_IN.in'
    config.test_results_dir = "RESULTS"
    config.name = 'Seg3D_n_case128'
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.low_parameter_path = ''

    config.model = 'Seg3D_low'
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

    config.save = False
    config.debug = True
    if config.debug:
        config.epochs = 1
        config.project = 'debug'
        config.n_case = 128
    else:
        config.epochs = 20
        config.project = 'lung'
        config.n_case = 128
    return config

def resample(img, hdr, new_spacing=[1,1,1], new_shape=None):
    # new_shape = (64,64,64)
    if new_shape is None:
        spacing = np.array(hdr.spacing, dtype=np.float32)
        resize_factor = spacing / new_spacing
        new_real_shape = img.shape * resize_factor
        new_shape = np.round(new_real_shape)

    real_resize_factor = np.array(new_shape) / img.shape
    img = scipy.ndimage.interpolation.zoom(img,real_resize_factor, mode='nearest')
    return img





    
if __name__ == "__main__": 
    parameter_path = '/data1/inqlee0704/lung_segmentation/RESULTS/Seg3D_n_case128_20210819/lung_Seg3D_low.pth'
    load_dotenv()
    seed_everything()
    config = wandb_config()
    
    # Data
    df_subjlist = pd.read_csv(os.path.join(config.data_path,config.in_file),sep='\t')
    df_train, df_valid = model_selection.train_test_split(
        df_subjlist[:config.n_case],
        test_size=0.2,
        random_state=42,
        stratify=None)
    train_ds = LungDataset_3D_low_res(df_train.reset_index(drop=True))
    train_loader = DataLoader(train_ds,
                                batch_size=config.train_bs,
                                shuffle=True,
                                num_workers=0)

    model = Seg3D(num_classes=1)
    model.load_state_dict(torch.load(parameter_path))
    model.to(config.device)
    model.eval()
    eng = Segmentor(model, device=config.device)
    batch_preds = eng.predict(train_loader)
    print(batch_preds)
    print(batch_preds[0].shape)


