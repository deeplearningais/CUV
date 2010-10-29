import numpy as np
import os
import sys
import pdb
from ais_functions import *
from scipy.io.numpyio import fwrite, fread

path = "./"

if len(sys.argv)>1:
    path=sys.argv[1]

print "loading weights from ",path
w=np.load(os.path.join(path,"weights-0.npy"))
i=np.load(os.path.join(path,"iweights-0.npy"))
#i=np.zeros((w.shape[1],w.shape[1])).astype('float32').copy('F')
x=np.load(os.path.join(path,"hidden_rep-0.npy"))
mu=x.mean(axis=1)
mu[-1]=1
fd=open(os.getenv("HOME")+'/MNIST/train-images.idx3-ubyte') 
fread(fd,16,'c')
data = np.fromfile(file=fd, dtype=np.uint8).reshape( (60000,784) )/255

v=np.ones((data.shape[0],data.shape[1]+1))
v[:,0:-1]=data

p_v=0
row=v
h=mu
pdb.set_trace()
p_v = np.dot(h,np.dot(i,h))+np.dot(row,np.dot(w,h)) - (mu[0:-1] *
                                                       np.log(mu[0:-1])+(1-mu[0:-1])*np.log(1-mu[0:-1])).sum() 
print "mean: ",p_v.mean()," std: ",p_v.std()
ut=p_v.mean
