import cuv_python as cp
import pyublas
import numpy as np
import os

def kmeans(dataset,num_clusters,iters):
    rand_indices=np.random.randint(0,dataset.shape[1],num_clusters)
    clusters=dataset[:,rand_indices]
    dataset_dev=cp.push(dataset)
    clusters_dev=cp.push(clusters.copy("F")) # copy('F') is necessary so we can slice later on

    norms = cp.dev_matrix_cmf(dataset_dev.w, 1)
    cp.reduce_to_row(norms.vec,dataset_dev,cp.reduce_functor.ADD_SQUARED)

    norms_clusters=cp.dev_matrix_cmf(num_clusters,1)
    dists  = cp.dev_matrix_cmf(dataset_dev.w, num_clusters)
    nearest= cp.dev_matrix_cmui(dataset_dev.w,1)
    nearest_dist= cp.dev_matrix_cmf(dataset_dev.w,1)

    for i in xrange(iters):
        cp.reduce_to_row(norms_clusters.vec,clusters_dev,cp.reduce_functor.ADD_SQUARED)
        cp.prod(dists, dataset_dev, clusters_dev, 't','n',-2, 0)
        cp.matrix_plus_row(dists,norms_clusters.vec)
        cp.matrix_plus_col(dists,norms.vec)
        cp.reduce_to_col(nearest.vec,dists,cp.reduce_functor.ARGMIN)
        cp.reduce_to_col(nearest_dist.vec,dists,cp.reduce_functor.MIN) # as stopping criterion
        #print("Average dist to cluster: %f"%cp.mean(nearest_dist))
        cp.compute_clusters(clusters_dev,dataset_dev,nearest.vec)
    return [clusters_dev.np, nearest.np]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    path="/home/local/datasets/MNIST"

    with open(os.path.join(path,'train-images.idx3-ubyte')) as fd:
        np.fromfile(file=fd,dtype=np.uint8,count=16)
        mnist = np.fromfile(file=fd, dtype=np.uint8).reshape( (60000,784)).astype('float32').T

    num_clusters=20

    [clusters, indices]=kmeans(mnist,num_clusters,10)

    for i in xrange(num_clusters):
        plt.subplot(np.ceil(num_clusters/5),5,i+1)
        plt.imshow(clusters[:,i].reshape(28,28))
    plt.show()
