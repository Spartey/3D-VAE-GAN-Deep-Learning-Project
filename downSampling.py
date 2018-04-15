import numpy as np
import matplotlib.pyplot as plt
from numpy import nanmean
import binvox_rw
from PIL import Image
import imageio


image_path = "./office-chair-image/d9c4ce130b7412bb1c1b3b2ed8d13bf8-6.png"
model_path = "./office-chair-model/1e6a212abb46d63df91663a74ccd2338.binvox"

# Method modified from https://github.com/keflavich/image_registration/blob/master/image_registration/fft_tools/downsample.py#L11

def downsample(myarr,factor,estimator=nanmean):
    ys,xs = myarr.shape
    crarr = myarr[:ys-(ys % int(factor)),:xs-(xs % int(factor))]
    dsarr = estimator( np.concatenate([[crarr[i::factor,j::factor]
        for i in range(factor)]
        for j in range(factor)]), axis=0)
    return dsarr


def downsample_cube(myarr,factor,ignoredim=0, estimator=nanmean):
    if ignoredim > 0: myarr = myarr.swapaxes(0,ignoredim)
    zs,ys,xs = myarr.shape
    crarr = myarr[:,:ys-(ys % int(factor)),:xs-(xs % int(factor))]
    dsarr = estimator(np.concatenate([[crarr[:,i::factor,j::factor]
        for i in range(factor)]
        for j in range(factor)]), axis=0)
    if ignoredim > 0: dsarr = dsarr.swapaxes(0,ignoredim)
    return dsarr


def downsample_cube2(myarr,factor,estimator=nanmean):
    zs,ys,xs = myarr.shape
    crarr = myarr[:zs-(zs % int(factor)), :ys-(ys % int(factor)), :xs-(xs % int(factor))]
    dsarr = estimator(np.concatenate([[[crarr[k::factor, i::factor, j::factor]
        for i in range(factor)]
        for j in range(factor)]
        for k in range(factor)]), axis=(0, 1))
    return dsarr

img = Image.open(image_path).convert('L')
img.save('./greyscale.png')
im = imageio.imread("./greyscale.png")
res = downsample(im, 8)
plt.imshow(res)
plt.show()

with open(model_path, 'rb') as f:
    m1 = binvox_rw.read_as_3d_array(f)

a = m1.data
res = downsample_cube2(a, 4)


