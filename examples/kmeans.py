import cuv_python as cp
import numpy as np
import os

def kmeans(dataset,num_clusters,iters):
    rand_indices=np.random.randint(0,dataset.shape[1],num_clusters)
    clusters=dataset[:,rand_indices]
    dataset_dev=cp.dev_tensor_float_cm(dataset)
    clusters_dev=cp.dev_tensor_float_cm(clusters.copy("F")) # copy('F') is necessary so we can slice later on

    norms = cp.dev_tensor_float(dataset_dev.shape[1])
    cp.reduce_to_row(norms,dataset_dev,cp.reduce_functor.ADD_SQUARED)

    dists = cp.dev_tensor_float([dataset_dev.shape[1], num_clusters])
    nearest = cp.dev_tensor_uint(dataset_dev.shape[1])

    for i in xrange(iters):
        cp.pdist2(dists, dataset_dev.T,clusters_dev.T)
        cp.reduce_to_col(nearest,dists,cp.reduce_functor.ARGMIN)
        cp.compute_clusters(clusters_dev,dataset_dev,nearest)
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
        plt.subplot(int(np.ceil(num_clusters/5)),5,i+1)
        plt.imshow(clusters[:,i].reshape(28,28))
    plt.show()
