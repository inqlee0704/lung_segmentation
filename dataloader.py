""" ****************************************** 
    Author: In Kyu Lee
    Deep learning dataloaders are stored here.
    Available:
    - ImageDataset: classification
    - SegDataset: Semantic segmentation
    - slice_loader: load slice information for SegDataset
    - CT_loader: load CT images
    - SlicesDataset: Semantic Segmentation (load all images into memory)
    - check_files: check if ct and mask file exist
****************************************** """ 
import os
from medpy.io import load
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection
import albumentations as A
from albumentations.pytorch import ToTensorV2

import scipy.ndimage

def resample(img, hdr, new_spacing=[1,1,1], new_shape=None):
    # new_shape = (64,64,64)
    if new_shape is None:
        spacing = np.array(hdr.spacing, dtype=np.float32)
        resize_factor = spacing / new_spacing
        new_real_shape = img.shape * resize_factor
        new_shape = np.round(new_real_shape)
    # new_spacing = spacing / real_resize_factor
    real_resize_factor = np.array(new_shape) / img.shape
    img = scipy.ndimage.interpolation.zoom(img,real_resize_factor, mode='nearest')
    return img, real_resize_factor

def upsample(img, real_resize_factor):
    img = scipy.ndimage.interpolation.zoom(img,1/real_resize_factor, mode='nearest')
    return img

class LungDataset_3D:
    def __init__(self, subjlist, slabs, low_model, augmentations=None):
        self.subj_paths = subjlist.loc[:,'ImgDir'].values
        self.img_paths = [os.path.join(subj_path,'zunu_vida-ct.img') for subj_path in self.subj_paths]
        self.mask_paths = [os.path.join(subj_path,'ZUNU_vida-lung.img.gz') for subj_path in self.subj_paths]
        self.img = None
        self.low_output = None
        self.mask = None
        self.slabs = slabs 
        self.ID = None
        self.low_model = low_model
        self.augmentations = augmentations

    def infer_low(self,img,hdr):
        # downsample
        img, resize_factor = resample(img,hdr,new_shape=(64,64,64))
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        output = self.low_model(img.to('cuda', dtype=torch.float))
        pred = torch.sigmoid(output)
        pred = pred.cpu().detach().numpy()
        pred = np.squeeze(pred)
        # upsample to the original dimension
        pred = upsample(pred,resize_factor)
        return pred

    def __len__(self):
        return len(self.slabs)

    def __getitem__(self,idx):
        # 256x256xz, z: 1mm slice thickness
        slab = self.slabs[idx]
        if self.ID != slab[0]:
            self.img, hdr = load(self.img_paths[slab[0]])
            self.img = (self.img-np.min(self.img))/(np.max(self.img)-np.min(self.img))
            # low resolution output
            self.low_output = self.infer_low(self.img,hdr)
            self.img, _ = resample(self.img,hdr,new_spacing=(hdr.spacing[0]*2,hdr.spacing[1]*2,1))
            self.low_output, _ = resample(self.low_output,hdr,new_spacing=(hdr.spacing[0]*2,hdr.spacing[1]*2,1))
            mask, hdr = load(self.mask_paths[slab[0]])
            mask[mask==20] = 1
            mask[mask==30] = 1
            self.mask, _ = resample(mask,hdr,new_spacing=(hdr.spacing[0]*2,hdr.spacing[1]*2,1))
            self.mask = self.mask.astype(int)
            # do padding for the last slab
            if mask.shape[2]%32 != 0:
                missing = np.ceil(self.mask.shape[2]/32)*32 - self.mask.shape[2]
                self.mask = np.concatenate([self.mask,np.zeros((256,256,int(missing)))],axis=2)
                self.img = np.concatenate([self.img,np.zeros((256,256,int(missing)))],axis=2)
                self.low_output = np.concatenate([self.low_output,np.zeros((256,256,int(missing)))],axis=2)

            self.ID = slab[0]

        img = self.img[:,:,slab[1]*32:slab[1]*32+32]
        low_output = self.low_output[:,:,slab[1]*32:slab[1]*32+32]
        mask = self.mask[:,:,slab[1]*32:slab[1]*32+32]
        img = torch.from_numpy(img).unsqueeze(0)
        low_output = torch.from_numpy(low_output).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        # combine
        img = torch.cat([img,low_output],axis=0)

        # img = img[None,:]
        # low_output = low_output[None,:]
        # mask = mask[None,:]
        if self.augmentations is not None:
            augmented = self.augmentations(image=img,mask=mask)
            img,mask = augmented['image'], augmented['mask']

        return {
                'image': img,
                'seg': mask
                }
        # return {
        #         'image': torch.tensor(img.copy()),
        #         'seg': torch.tensor(mask.copy())
        #         }



class LungDataset_3D_high_res:
    def __init__(self, subjlist, slabs, augmentations=None):
        self.subj_paths = subjlist.loc[:,'ImgDir'].values
        self.img_paths = [os.path.join(subj_path,'zunu_vida-ct.img') for subj_path in self.subj_paths]
        self.mask_paths = [os.path.join(subj_path,'ZUNU_vida-lung.img.gz') for subj_path in self.subj_paths]
        self.img = None
        self.mask = None
        self.slabs = slabs 
        self.ID = None
        self.augmentations = augmentations

    def __len__(self):
        return len(self.slabs)

    def __getitem__(self,idx):
        # 256x256xz, z: 1mm slice thickness
        slab = self.slabs[idx]
        if self.ID != slab[0]:
            img, hdr = load(self.img_paths[slab[0]])
            self.img, _ = resample(img,hdr,new_spacing=(hdr.spacing[0]*2,hdr.spacing[1]*2,1))
            self.img = (self.img-np.min(self.img))/(np.max(self.img)-np.min(self.img))
            mask, hdr = load(self.mask_paths[slab[0]])
            mask[mask==20] = 1
            mask[mask==30] = 1
            self.mask, _ = resample(mask,hdr,new_spacing=(hdr.spacing[0]*2,hdr.spacing[1]*2,1))
            self.mask = self.mask.astype(int)
            # do padding for the last slab
            if mask.shape[2]%32 != 0:
                missing = np.ceil(self.mask.shape[2]/32)*32 - self.mask.shape[2]
                self.mask = np.concatenate([self.mask,np.zeros((256,256,int(missing)))],axis=2)
                self.img = np.concatenate([self.img,np.zeros((256,256,int(missing)))],axis=2)
            self.ID = slab[0]

        img = self.img[:,:,slab[1]*32:slab[1]*32+32]
        mask = self.mask[:,:,slab[1]*32:slab[1]*32+32]
        img = img[None,:]
        mask = mask[None,:]
        if self.augmentations is not None:
            augmented = self.augmentations(image=img,mask=mask)
            img,mask = augmented['image'], augmented['mask']
        return {
                'image': torch.tensor(img.copy()),
                'seg': torch.tensor(mask.copy())
                }

class LungDataset_3D_low_res:
    def __init__(self, subjlist, augmentations=None, test=False):
        self.subj_paths = subjlist.loc[:,'ImgDir'].values
        self.img_paths = [os.path.join(subj_path,'zunu_vida-ct.img') for subj_path in self.subj_paths]
        self.mask_paths = [os.path.join(subj_path,'ZUNU_vida-lung.img.gz') for subj_path in self.subj_paths]
        self.img = None
        self.mask = None
        self.augmentations = augmentations
        self.test = test

    def __len__(self):
        return len(self.subj_paths)

    def __getitem__(self,idx):
        img, hdr = load(self.img_paths[idx])
        im, _ = resample(img,hdr,new_shape=(64,64,64))
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        img = img[None,:]
        if self.test: # only return images
            return {'image': torch.tensor(img.copy())}

        mask, hdr = load(self.mask_paths[idx])
        mask[mask==20] = 1
        mask[mask==30] = 1
        mask, _ = resample(mask,hdr,new_shape=(64,64,64))
        mask = mask.astype(int)
        mask = mask[None,:]
        if self.augmentations is not None:
            augmented = self.augmentations(image=img,mask=mask)
            img,mask = augmented['image'], augmented['mask']

        return {
                'image': torch.tensor(img.copy()),
                'seg': torch.tensor(mask.copy())
                }

"""
ImageDataset for 3D CT image Segmentation
Inputs:
    - subjlist: panda's dataframe which contains image & mask paths [df]
    - slices: slice information from slice_loader function [list]
Outputs:
    - dictionary that containts both image tensor & mask tensor [dict]
"""
class SegDataset:
    def __init__(self,subjlist, slices, mask_name=None,
                 resize=None, augmentations=None):
        self.subj_paths = subjlist.loc[:,'ImgDir'].values
        self.img_paths = [os.path.join(subj_path,'zunu_vida-ct.img') for subj_path in self.subj_paths]
        self.mask_paths = [os.path.join(subj_path,'ZUNU_vida-airtree.img.gz') for subj_path in self.subj_paths]
        self.slices = slices
        self.pat_num = None
        self.img = None
        self.mask = None
        self.mask_name = mask_name
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.slices)

    def __getitem__(self,idx):
        slc = self.slices[idx]
        if self.pat_num != slc[0]:
            self.img,_ = load(self.img_paths[slc[0]])
            self.mask,_ = load(self.mask_paths[slc[0]])
            self.pat_num = slc[0]
        img = self.img[:,:,slc[1]]
        mask = self.mask[:,:,slc[1]]
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        # img = (img-(-1250))/((250)-(-1250))
        # Airway mask is stored as 255
        if self.mask_name=='airway':
            mask = mask/255
        elif self.mask_name=='lung':
            mask[mask==20] = 1
            mask[mask==30] = 1
        else:
            print('Specify mask_name (airway or lung)')
            return -1
        mask = mask.astype(int)
        img = img[None,:]
        mask = mask[None,:]
        if self.resize is not None:
            img = cv2.resize(img,
                            (self.resize[1], self.resize[0]),
                             interpolation=cv2.INTER_CUBIC)
        if self.augmentations is not None:
            augmented = self.augmentations(image=img,mask=mask)
            img,mask = augmented['image'], augmented['mask']

        return {
                'image': torch.tensor(img.copy()),
                'seg': torch.tensor(mask.copy())
                }

"""
Slab loader which outputs slice information for each CT
and check if mask dimension and image dimension match
Inputs:
    - subjlist: panda's dataframe which contains image & mask paths [df]
Outputs:
    - A slice list of tuples, [list]
        - first index represent subject's number
        - second index represent axial position of CT
    ex) (0,0),(0,1),(0,2) ... (0,750),(1,0),(1,1) ... (300, 650)
"""
def slab_loader(subjlist):
    print('Loading Data')
    subj_paths = subjlist.loc[:,'ImgDir'].values
    img_paths = [os.path.join(subj_path,'zunu_vida-ct.img') for subj_path in subj_paths]
    mask_paths = [os.path.join(subj_path,'ZUNU_vida-lung.img.gz') for subj_path in subj_paths]
    slabs = []
    slab_size = 32
    for ii in range(len(mask_paths)):
        label,_ = load(mask_paths[ii])
        img,hdr = load(img_paths[ii])
        num_slices = hdr.spacing[2] * img.shape[2]
        num_slabs = np.ceil(num_slices/slab_size)
        if img.shape != label.shape:
            print('Dimension does not match: ')
            print(subjlist.loc[ii,'ImgDir'])
        for jj in range(int(num_slabs)):
            slabs.append((ii,jj))
    return slabs


"""
Prepare train & valid dataloaders
"""
def prep_dataloader(c,n_case=0,LOAD_ALL=False):
# n_case: load n number of cases, 0: load all
    df_subjlist = pd.read_csv(os.path.join(c.data_path,c.in_file),sep='\t')
    if n_case==0:
        df_train, df_valid = model_selection.train_test_split(
                df_subjlist,
                test_size=0.2,
                random_state=42,
                stratify=None)
    else:
        df_train, df_valid = model_selection.train_test_split(
             df_subjlist[:n_case],
             test_size=0.2,
             random_state=42,
             stratify=None)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    if LOAD_ALL:
        train_loader = DataLoader(SlicesDataset(CT_loader(df_train,
            mask_name=c.mask)),
            batch_size=c.train_bs, 
            shuffle=True,
            num_workers=0)
        valid_loader = DataLoader(SlicesDataset(CT_loader(df_valid,
            mask_name=c.mask)),
            batch_size=c.valid_bs, 
            shuffle=True,
            num_workers=0)
    else:
        train_slices = slice_loader(df_train)
        valid_slices = slice_loader(df_valid)
        train_ds = SegDataset(df_train,
                              train_slices,
                              mask_name=c.mask,
                              augmentations=get_train_aug())
        valid_ds = SegDataset(df_valid, valid_slices, mask_name=c.mask)
        train_loader = DataLoader(train_ds,
                                  batch_size=c.train_bs,
                                  shuffle=False,
                                  num_workers=0)
        valid_loader = DataLoader(valid_ds,
                                  batch_size=c.valid_bs,
                                  shuffle=False,
                                  num_workers=0)

    return train_loader, valid_loader

def get_train_aug():
    return A.Compose([
        A.Rotate(limit=10),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ],p=0.5),
        A.OneOf([
            A.Blur(blur_limit=5),
            A.MotionBlur(blur_limit=7),
            A.GaussianBlur(blur_limit=(3,7)),
        ],p=0.5),
        # ToTensorV2()
    ])

# def get_valid_aug():
#     return A.Compose([
#         A.OneOf([
#             A.HorizontalFlip(),
#             A.VerticalFlip(),
#         ],p=0.4),
#     ])
