#!/usr/bin/env python

import Image
import pyublas
import numpy as np
import cuv_python as cp
import matplotlib.pyplot as plt
from timeit import Timer


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

def testbuildpyra(pic,input_channels,pyramid_channels):
    pic_d = cp.push(pic)
    pyr = cp.dev_image_pyramid_f(pic_d.h/2,pic_d.w/input_channels/2,4,pyramid_channels)
    pyr.build(pic_d,4)
    #plt.matshow(cp.pull(pyr.get(0,0)))
    #plt.matshow(cp.pull(pyr.get(0,1)))
    #plt.matshow(cp.pull(pyr.get(0,2)))
    #plt.show()


def test_cuda_array(pic):
    pic_h = cp.push_host(pic)

    # downsample pic_h
    tmp = (np.array([pic_h.h,pic_h.w])/2).astype('uint32')
    down  = cp.dev_matrix_rmf(tmp[0],tmp[1])
    ca    = cp.dev_cuda_array_f(pic_h.h,pic_h.w)
    ca.assign(pic_h)
    cp.gaussian_pyramid_downsample(down,ca)

    # upsample downsampled
    ca    = cp.dev_cuda_array_f(down.h,down.w)
    ca.assign(down)
    up    = cp.dev_matrix_rmf(pic_h.h, pic_h.w)
    cp.gaussian_pyramid_upsample(up,ca)
    print cp.pull(up)

    plt.matshow(cp.pull(down))
    plt.title("Downsampled")
    plt.matshow(pic)
    plt.title("Original")
    plt.matshow(cp.pull(up))
    plt.title("Upsampled")
    plt.show()
    ca.dealloc()

def run():
    pic = Image.open("tests/data/colored_square.jpg").resize((128,128)).convert("RGBA")
    pig = Image.open("tests/data/gray_square.gif").resize((128,128)).convert("L")
    #color_test(np.asarray(pic).reshape(128**2*4,1))
    #gray_test( np.asarray(pig).reshape(128**2  ,1))

    pig = Image.open("tests/data/gray_square.gif").resize((640,480)).convert("L")
    #test_cuda_array(np.asarray(pig).astype("float32"))


if __name__ == "__main__":
    cp.initCUDA(0)

    #pic = Image.open("tests/data/gray_square.gif").resize((1024,768)).convert("RGB")
    #pic = np.asarray(pic).astype("float32")
    pic = Image.open("tests/data/gray_square.gif").resize((1024,768)).convert("RGBA")
    pic = np.asarray(pic).astype("float32").reshape(768,1024*4)
    for x in xrange(1): # warmup
        testbuildpyra(pic,input_channels=4,pyramid_channels=3)

    t = Timer('testbuildpyra(pic,4,3)','from %s import testbuildpyra, cp, pic'%__name__)
    print t.timeit(number=1000)/1000

    run()

    cp.exitCUDA()
