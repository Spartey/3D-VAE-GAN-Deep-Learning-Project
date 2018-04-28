import numpy as np
import matplotlib.pyplot as plt
from numpy import nanmean
from PIL import Image
import imageio
import importlib

# Method modified from https://github.com/keflavich/image_registration/blob/master/image_registration/fft_tools/downsample.py#L11


binvox_rw = importlib.import_module("binvox_rw")


def downsample(myarr,factor,estimator=nanmean):
    ys,xs = myarr.shape
    crarr = myarr[:ys-(ys % int(factor)),:xs-(xs % int(factor))]
    dsarr = estimator( np.concatenate([[crarr[i::factor,j::factor]
        for i in range(factor)]
        for j in range(factor)]), axis=0)
    return dsarr


def downsample_cube(myarr,factor,estimator=nanmean):
    zs,ys,xs = myarr.shape
    crarr = myarr[:zs-(zs % int(factor)), :ys-(ys % int(factor)), :xs-(xs % int(factor))]
    dsarr = estimator(np.concatenate([[[crarr[k::factor, i::factor, j::factor]
        for i in range(factor)]
        for j in range(factor)]
        for k in range(factor)]), axis=(0, 1))
    return dsarr

