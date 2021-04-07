# inference.py
# ##############################################################################
# 20210312, In Kyu Lee
# Desc: Deep Learning Inference
# ##############################################################################
# Functions:
#  - run_inference
#  - volume_inference
# ##############################################################################
# How to use:
# run_inference(root_path,parameter_path)
# ##############################################################################
import os
from model_util import RecursiveUNet
import time
import torch
from medpy.io import load, save
import numpy as np
import pandas as pd
import nibabel as nib
import shutil # copy file
import gzip # unzip .hdr.gz -> .hdr


# Semantic Segmentation infercence
def run_inference(root_path,parameter_path):
    print('Data Loading . . .')
    img_path = os.path.join(root_path,'zunu_vida-ct.img')
    image,hdr = load(img_path)
    image = (image-(np.min(image)))/((np.max(image)-(np.min(image))))
    out = []
    out.append({'image':image})
    test_data = np.array(out)
    model = RecursiveUNet()
    model.load_state_dict(torch.load(parameter_path))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(DEVICE)
    model.eval()
    print('Start Inference . . .')
    for x in test_data:
        pred_label = volume_inference(model,x['image'])
        pred_label = pred_label * 255
        pred_label = pred_label.astype(np.ubyte)
        save(pred_label,os.path.join(root_path,'ZUNU_unet-airtree.img.gz'),hdr=hdr)

    with gzip.open(os.path.join(root_path,'ZUNU_unet-airtree.hdr.gz'),'rb') as f_in:
        with open(os.path.join(root_path,'ZUNU_unet-airtree.hdr'),'wb') as f_out:
            shutil.copyfileobj(f_in,f_out)
    os.remove(os.path.join(root_path,'ZUNU_unet-airtree.hdr.gz'))
    

def volume_inference(model, volume):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    slices = np.zeros(volume.shape)
    for i in range(volume.shape[2]):
        s = volume[:,:,i]
        s = s.astype(np.single)
        s = torch.from_numpy(s).unsqueeze(0).unsqueeze(0)
        pred = model(s.to(DEVICE))
        pred = np.squeeze(pred.cpu().detach())
        slices[:,:,i] = torch.argmax(pred,dim=0)
    return slices
