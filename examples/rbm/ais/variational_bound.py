from numpy import *
import pdb
import sys
import os
from scipy.io.numpyio import fwrite, fread

def sample(x):
    noise = (random.uniform(0,1,x.shape))
    y = (noise < x)+0
    return y
def sigm(x):
    return 1/(1+exp(-x))
def p(v,h):
    ### lower level rbm p(v|h)
    #p_v_h = exp(dot(v[0:-1].T,dot(w0,h)[0:-1,:]))/ ( (1+exp(dot(w0[0:-1,:],h[:,:]))).prod(axis=0) * exp(dot(w0[0:-1,-1].T,v[0:-1,:])))
    p_v_h = exp(dot(v.T,dot(w0,h)))/  ((1+exp(dot(w0[0:-1,:],h[:,:]))).prod(axis=0)*  exp(dot(w0[-1,0:-1],h[0:-1,:])))
    ### prior on p(h) via higher level rbm
    p_h = log((1+exp(dot(w1[:,0:-1].T,h[:,:])))).sum(axis=0) + (dot(w1[0:-1,-1],h[0:-1,:]))
    #print "log(p(h)): ", log(p_h), " log(p(v|h)): ", log(p_v_h), " log (p(v)) ",log(p_v_h*p_h)
    return log(p_v_h) + p_h

def entropy(v):
    ### lower level rbm p(h|1)
    ### sense, the next line makes none
    ###p_h_v = exp(dot(h[0:-1].T,dot(w0.T,v)[0:-1,:])) / ((1+exp(dot(w0[0:-1,0:-1],h[0:-1,:]))).prod(axis=0) * exp(dot(w0[-1,0:-1],h[0:-1,:])))
    #return - p_h_v * log(p_h_v)
    vW = dot(w0.T,v)[0:-1,:]
    tmp = vW/(1+exp(-vW))-log(1+exp(vW))
    return -tmp.sum(axis=0)


path = "."
if len(sys.argv)>1:
    path=sys.argv[1]

### load mnist
fd = open(os.getenv("HOME")+'/MNIST/t10k-images.idx3-ubyte')
#fd = open(os.getenv("HOME")+'/MNIST/train-images.idx3-ubyte')
fread(fd,16,'c')
data = fromfile(file=fd, dtype=uint8).reshape( (10000,784) )
#data = fromfile(file=fd, dtype=uint8).reshape( (60000,784) )

### load weights
print "loading weights from ",path
w0=load(os.path.join(path,"weights-0.npy"))
w1=load(os.path.join(path,"weights-1.npy"))

data = data[0:1000,:].T
data = data.astype('float')/255
v_org = ones((data.shape[0]+1,data.shape[1]))
v_org[0:-1,:]=data

datalength=data.shape[1]
batchsize=1
M=5
#est = zeros(batchsize)
out=0
### iterate over batches
for n in xrange(datalength/batchsize):
    v=v_org[:,n*batchsize:(n+1)*batchsize]
    est =0
    for step in xrange(M):
         h = sample(sigm(dot(w0.T , v)))
         h[-1,:]=1 ## reset bias
         # visualize
         if n % 100 == 0:
             import matplotlib.pyplot as plt
             plt.figure(1)
             plt.matshow(sigm(dot(w0,h)[0:784,0].reshape((28,28))))
             plt.draw()
             plt.savefig("chain_variational_%05d.png"%n)
         asdf = p(v,h)
         est += asdf
    est = est/M
    entr= entropy(v)
    out += est  + entr
    if n%100 == 0:
        #print "est: ",est, "out: ", out/(n+1), "entropy ", entr
        sys.stdout.write('.')
        sys.stdout.flush()

out = out/(datalength/batchsize)
print ""
print "variational lower bound: ", out 
