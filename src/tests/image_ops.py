#!/usr/bin/env python

import Image
import pyublas
import numpy as np
import cuv_python as cp
import matplotlib.pyplot as plt
from ipdb import set_trace


def to_cmuc(x):
    if x.dtype != np.dtype("uint8"):
        x = x.astype('uint8')
    if not x.flags.fortran:
        x = x.copy('F')
    return x


def gray_test(ni):
    src = cp.push(to_cmuc(np.tile(ni,(1,4))))
    dst = cp.dev_matrix_cmf(src.h,src.w)
    cp.fill(dst,0)
    cp.image_move(dst,src,128,128,1,-10,-4)
    res = cp.pull(dst)
    #set_trace()
    plt.matshow(res[0:128**2,0].reshape(128,128))
    plt.colorbar()
    plt.show()

def color_test(ni):
    ts = 128
    src = cp.push(to_cmuc(np.tile(ni,(1,4))))
    dst = cp.dev_matrix_cmf(ts**2*3,src.w)
    cp.fill(dst,0)
    cp.image_move(dst,src,128,ts,4,-10,-4)
    res = cp.pull(dst)
    plt.matshow(res[0:ts**2,0].reshape(ts,ts), cmap = plt.cm.bone_r)
    plt.matshow(res[ts**2:2*ts**2,0].reshape(ts,ts), cmap = plt.cm.bone_r)
    plt.matshow(res[2*ts**2:3*ts**2,0].reshape(ts,ts), cmap = plt.cm.bone_r)
    plt.show()

def test_cuda_array(pic):
    pic_h = cp.push_host(pic)
    tmp = (np.array([pic_h.h,pic_h.w])/2).astype('uint32')
    down  = cp.dev_matrix_rmf(tmp[0],tmp[1])
    ca    = cp.dev_cuda_array_f(pic_h.h,pic_h.w)
    ca.assign(pic_h)
    cp.gaussian_pyramid_downsample(down,ca)
    x     = cp.pull(down)
    print x
    plt.matshow(x)
    plt.matshow(pic)
    plt.show()
    ca.dealloc()

def run():
    cp.initCUDA(0)
    pic = Image.open("tests/data/colored_square.jpg").resize((128,128)).convert("RGBA")
    pig = Image.open("tests/data/gray_square.gif").resize((128,128)).convert("L")
    #color_test(np.asarray(pic).reshape(128**2*4,1))
    #gray_test( np.asarray(pig).reshape(128**2  ,1))
    test_cuda_array(np.asarray(pig).astype("float32"))
    cp.exitCUDA()

if __name__ == "__main__":
    run()
