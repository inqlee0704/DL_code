# model_util.py
# ##############################################################################
# 20210308, In Kyu Lee
# Desc: Deep Learning metrics
# ##############################################################################
# Segmentation:
#  - Dice3d(a,b)
# ##############################################################################
# How to use:
#
# ##############################################################################

import numpy as np

def Dice3d(a,b):
    intersection =  np.sum((a!=0)*(b!=0))
    volume = np.sum(a!=0) + np.sum(b!=0)
    if volume == 0:
        return -1
    return 2.*float(intersection)/float(volume)