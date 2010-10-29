import numpy as np
import os
import pdb
from scipy.io.numpyio import fwrite, fread
from ais_functions import *
import sys
path = "./"

if len(sys.argv)>1:
    path=sys.argv[1]

w=np.load(os.path.join(path,"weights-0.npy"))
w2=np.load(os.path.join(path,"weights-1.npy"))
x=np.load(os.path.join(path,"hidden_rep-mnist-0.npy"))
for a in xrange(1):
    x[-1,:]=1
    h2=sigm(np.dot(w2.T,x))
    h2[-1,:]=1
    x=sigm(np.dot(w2,h2))
mu=x.mean(axis=1)
mu[-1]=1

fd=open(os.getenv("HOME")+'/MNIST/train-images.idx3-ubyte') 
fread(fd,16,'c')
data = np.fromfile(file=fd, dtype=np.uint8).reshape( (60000,784) )/255
pdb.set_trace()
v=np.ones((data.shape[0],data.shape[1]+1))
v[:,0:-1]=data
p_h= np.log(1+np.exp(np.dot(w2.T[0:-1,:],mu))).sum()+np.dot(w2.T[-1,0:-1],mu[0:-1])
p_v_h=np.dot(v,np.dot(w,mu))-(np.log(1+np.exp(np.dot(w[0:-1,:],mu))).sum() +np.dot(w[-1,:],mu))
H_h = - (mu[0:-1] * np.log(mu[0:-1])+(1-mu[0:-1])*np.log(1-mu[0:-1])).sum() 
p_v=p_h+p_v_h + H_h
print "mean: ",p_v.mean()," std: ",p_v.std()

