import cuv_python as cp
import pyublas
import numpy as np
import scipy.cluster.vq as vq
#from scipy.io.numpyio import fwrite, fread
import ipdb
import os
import matplotlib.pyplot as plt
import sys
import time

sys.excepthook = __IPYTHON__.excepthook

path="/home/local/datasets/MNIST"

with open(os.path.join(path,'train-images.idx3-ubyte')) as fd:
    np.fromfile(file=fd,dtype=np.uint8,count=16)
    mnist = np.fromfile(file=fd, dtype=np.uint8).reshape( (60000,784)).astype('float32').T
    

# start with 10 random samples
num_clusters=10

start_time=time.time()

#clusters=vq.kmeans(mnist.T,num_clusters,iter=10)
print("Scipy time: %f"%(time.time()-start_time))

start_time=time.time()
rand_indices=np.random.randint(0,mnist.shape[1],num_clusters)
clusters=mnist[:,rand_indices]
mnist_dev=cp.push(mnist)
clusters_dev=cp.push(clusters.copy("F")) # copy('F') is necessary so we can slice later on

norms = cp.dev_matrix_cmf(mnist_dev.w, 1)
cp.reduce_to_row(norms.vec,mnist_dev,cp.reduce_functor.ADD_SQUARED)

norms_clusters=cp.dev_matrix_cmf(num_clusters,1)
dists  = cp.dev_matrix_cmf(mnist_dev.w, num_clusters)
nearest= cp.dev_matrix_cmf(mnist_dev.w,1)
nearest_dist= cp.dev_matrix_cmf(mnist_dev.w,1)

for i in xrange(10):
    cp.reduce_to_row(norms_clusters.vec,clusters_dev,cp.reduce_functor.ADD_SQUARED)
    cp.prod(dists, mnist_dev, clusters_dev, 't','n',-2, 0)
    cp.matrix_plus_row(dists,norms_clusters.vec)
    cp.matrix_plus_col(dists,norms.vec)
    cp.reduce_to_col(nearest.vec,dists,cp.reduce_functor.ARGMIN)
    cp.reduce_to_col(nearest_dist.vec,dists,cp.reduce_functor.MIN) # as stopping criterion
    #print("Average dist to cluster: %f"%cp.mean(nearest_dist))
    nearest_host = nearest.np
    for j in xrange(num_clusters):
        indices = nearest_host==j
        indices_dev=cp.push(indices)
        tmp=mnist_dev.copy()
        cp.matrix_times_row(tmp,indices_dev.vec)
        mean_dev=cp.dev_matrix_cmf(mnist_dev.h,1)
        cp.reduce_to_col(mean_dev.vec,tmp)
        mean_dev*=1./indices.sum()
        clusters_dev[:,j]=mean_dev
    bla=cp.push(nearest_host.astype(np.uint32))
    test_clusters=clusters_dev.copy()
    cp.compute_clusters(test_clusters,mnist_dev,bla.vec)
cp.pull(clusters_dev)
print("CUV naive time: %f"%(time.time()-start_time))

start_time=time.time()
rand_indices=np.random.randint(0,mnist.shape[1],num_clusters)
clusters=mnist[:,rand_indices]
mnist_dev=cp.push(mnist)
clusters_dev=cp.push(clusters.copy("F")) # copy('F') is necessary so we can slice later on

norms = cp.dev_matrix_cmf(mnist_dev.w, 1)
cp.reduce_to_row(norms.vec,mnist_dev,cp.reduce_functor.ADD_SQUARED)

norms_clusters=cp.dev_matrix_cmf(num_clusters,1)
dists  = cp.dev_matrix_cmf(mnist_dev.w, num_clusters)
nearest= cp.dev_matrix_cmui(mnist_dev.w,1)
nearest_dist= cp.dev_matrix_cmf(mnist_dev.w,1)

for i in xrange(10):
    cp.reduce_to_row(norms_clusters.vec,clusters_dev,cp.reduce_functor.ADD_SQUARED)
    cp.prod(dists, mnist_dev, clusters_dev, 't','n',-2, 0)
    cp.matrix_plus_row(dists,norms_clusters.vec)
    cp.matrix_plus_col(dists,norms.vec)
    cp.reduce_to_col(nearest.vec,dists,cp.reduce_functor.ARGMIN)
    cp.reduce_to_col(nearest_dist.vec,dists,cp.reduce_functor.MIN) # as stopping criterion
    #print("Average dist to cluster: %f"%cp.mean(nearest_dist))
    cp.compute_clusters(clusters_dev,mnist_dev,nearest.vec)
cp.pull(clusters_dev)
print("CUV nice time: %f"%(time.time()-start_time))

clusters=clusters_dev.np
for i in xrange(num_clusters):
    plt.subplot(np.ceil(num_clusters/5),5,i+1)
    plt.imshow(clusters[:,i].reshape(28,28))
plt.show()

#ipdb.set_trace()



