# model_util.py
# ##############################################################################
# 20210308, In Kyu Lee
# Desc: Deep Learning metrics
# ##############################################################################
# Segmentation:
#  - Dice3d(a,b)
#  - Jaccard3d(a,b)
#  - Sensitivity(gt,pred)
#  - specificity(gt,pred)
# ##############################################################################

import numpy as np

def Dice3d(a,b):
    intersection =  np.sum((a!=0)*(b!=0))
    volume = np.sum(a!=0) + np.sum(b!=0)
    if volume == 0:
        return -1
    return 2.*float(intersection)/float(volume)

def Jaccard3d(a, b):
    intersection = np.sum((a!=0)*(b!=0))
    volumes = np.sum(a!=0) + np.sum(b!=0)
    if volumes == 0:
      return -1
    return float(intersection)/(float(volumes)-float(intersection))

def Sensitivity(gt,pred):
    # Sens = TP/(TP+FN)
    tp = np.sum(gt[gt==pred])
    fn = np.sum(gt[gt!=pred])
    if tp+fn==0:
        return -1
    return tp/(tp+fn)

def Specificity(gt,pred):
    # Spec = TN/(TN + FP)
    tn = np.sum((gt==0)&(pred==0))
    fp = np.sum((gt==0)&(pred!=0))
    if tn+fp == 0:
      return -1
    return tn/(tn+fp)