#!/usr/bin/env python

import Image
import pyublas
import numpy as np
import cuv_python as cp
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
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

def build_pyramid_GPU(pic,input_channels,pyramid_channels):
    pic_d = cp.push(pic)
    pyr = cp.dev_image_pyramid_f(pic_d.h/2,pic_d.w/input_channels/2,4,pyramid_channels)
    pyr.build(pic_d,4)

def build_pyramid_CPU(pic):
    L = [pic]
    for i in xrange(4):
        pic = gaussian_filter(pic,1)
        pic = pic[::2,::2].copy()
        L.append(pic)

def smooth(pic):
    ca = cp.dev_cuda_array_f(pic.h,pic.w,1)
    ca.assign(pic)
    cp.gaussian(pic,ca)
    ca.dealloc()
def test_pixel_classes():
    w, h = 512,512
    input_channels, pyramid_channels = 4,3
    pic = Image.open("tests/data/lena.bmp").resize((w,h)).convert("RGBA")
    pic = np.asarray(pic).astype("float32").reshape(h,w*4)
    pic_d = cp.push(pic)
    pyr = cp.dev_image_pyramid_f(pic_d.h/2,pic_d.w/input_channels/2,4,pyramid_channels)
    pyr.build(pic_d,4)
    plt.matshow(pic[0:h:2,0:4*w:8])
    #plt.matshow(cp.pull(pyr.get(1,0)))
    #plt.title("Channel0")
    #plt.matshow(cp.pull(pyr.get(1,1)))
    #plt.title("Channel1")
    #plt.matshow(cp.pull(pyr.get(1,2)))
    #plt.title("Channel2")
    #plt.matshow(cp.pull(pyr.get_all_channels(1)))
    #plt.title("allchannels level 1")
    #plt.show()

    # create source image from higher level of pyramid
    pic1 = pyr.get_all_channels(0)
    for i in xrange(10): smooth(pic1)
    plt.matshow(cp.pull(pic1)[:h/2,:w])
    ca = cp.dev_cuda_array_f(pic1.h,pic1.w,1)
    ca.assign(pic1)

    # create destination matrix
    pic0 = pyr.get(0)
    dst = cp.dev_matrix_rmuc(pic0.h,pic0.w*4) # uchar4

    # generate pixel classes and visualize
    cp.get_pixel_classes(dst,ca,1)
    tmp = cp.pull(dst)
    tmp = Image.frombuffer("CMYK", (pic0.w,pic0.h), cp.pull(dst).flatten(), "raw", "CMYK", 0, 1 ).resize((2*512,2*512), Image.NEAREST)
    tmp.show()
    print cp.pull(dst)
    plt.show()


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

def test_gaussian_pyramid_construction():
    print "Color Pyramid Construction on GPU: ",
    global pic
    w, h = 640,480
    pic = Image.open("tests/data/gray_square.gif").resize((w,h)).convert("RGBA")
    pic = np.asarray(pic).astype("float32").reshape(h,w*4)
    for x in xrange(1): # warmup
       build_pyramid_GPU(pic,input_channels=4,pyramid_channels=3)
    t = Timer('build_pyramid_GPU(pic,4,3)','from %s import build_pyramid_GPU, cp, pic'%__name__)
    print t.timeit(number=100)/100

    print "Grayscale Pyramid Construction on CPU: ",
    pic = Image.open("tests/data/gray_square.gif").resize((w,h)).convert("RGB")
    pic = np.asarray(pic).astype("float32")
    t = Timer('build_pyramid_CPU(pic)','from %s import build_pyramid_CPU, pic'%__name__)
    print t.timeit(number=100)/100


if __name__ == "__main__":
    cp.initCUDA(0)

    run()
    test_pixel_classes()
    test_gaussian_pyramid_construction()

    cp.exitCUDA()
