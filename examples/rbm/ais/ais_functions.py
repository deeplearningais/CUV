import inspect
import traceback
import sys
import os
import math
import pyublas
import cPickle

from scipy.io.numpyio import fwrite, fread
import numpy as np
import scipy
import pdb

def sigm(x):
    return 1/(1+np.exp(-x))
def sample(x,unit_type):

    if unit_type=="gaussian":
        y = (x + np.random.normal(0,1,x.shape)).astype('float32')
    else:
        noise = (np.random.uniform(0,1,x.shape)).astype('float32')
        y = (noise < x).astype("float32")
    return y

def binarize(x):
    if patches:
        return x
    else:
        return x>0.5 +0

#def load_patches():
    #data = np.load(os.path.join(os.getenv("HOME"), "prog/patches/patches-60000-28-28.npy")).T
    #data = np.log(1+data)
    #np.random.shuffle(data)

    #print "Loading patches, making 0-mean, unit-variance"
    #pcan = mdp.parallel.ParallelWhiteningNode(output_dim=data.shape[1]/4,
                                   #input_dim=data.shape[1], svd=True,
                                   #dtype='double')
    #pcan.train(data[0:10000,:])
    #pcan.stop_training()
    ##print "Eigenvalues: ", pcan.d
    #data = pcan.execute(data.astype('float32')) # whiten training data
    #data = np.dot(data, (pcan.v*np.sqrt(pcan.d)).T)   # transform back using whitened rotation matrix
    #data_mean2 = np.mean(data,axis=0) # 0-mean
    ##data_mean2[:] = 0
    #data -= data_mean2
    #data_std = np.std(data,axis=0)    # 1-variance
    ##data_std[:] = 1
    #data_std[data_std<0.01] = 0.01
    #data = data / data_std
    #return data.T

def load_mnist(training):
    if training:
        fd = open('/home/local/datasets/MNIST/train-images.idx3-ubyte')
    else:
        fd = open('/home/local/datasets/MNIST/t10k-images.idx3-ubyte')
    fread(fd,16,'c')
    if training:
        data = np.fromfile(file=fd, dtype=np.uint8).reshape( (60000,784) )
    else:
        data = np.fromfile(file=fd, dtype=np.uint8).reshape( (10000,784) )

    data = data[0:10000,:].T
    return data.astype('float')/255

