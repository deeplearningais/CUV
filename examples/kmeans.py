import cuv_python as cp
import numpy as np
import os

def kmeans(dataset,num_clusters,iters):
    # initialize clusters randomly
    rand_indices=np.random.randint(0,dataset.shape[0],num_clusters)
    clusters=dataset[rand_indices,:]

    # push initial clusters and dataset to device
    dataset_dev=cp.dev_tensor_float(dataset)
    clusters_dev=cp.dev_tensor_float(clusters)

    # allocate matrices for calculations (so we don't need to allocate in loop)
    dists = cp.dev_tensor_float([dataset_dev.shape[0], num_clusters])
    nearest = cp.dev_tensor_uint(dataset_dev.shape[0])

    # main loop
    for i in xrange(iters):
        # compute pairwise distances
        cp.pdist2(dists, dataset_dev, clusters_dev)
        # find closest cluster
        cp.reduce_to_col(nearest, dists, cp.reduce_functor.ARGMIN)
        # update cluster centers (this is a special purpose function for kmeans)
        cp.compute_clusters(clusters_dev, dataset_dev, nearest)
    return [clusters_dev.np, nearest.np]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    path="/home/local/datasets/MNIST"

    with open(os.path.join(path,'train-images.idx3-ubyte')) as fd:
        np.fromfile(file=fd,dtype=np.uint8,count=16)
        mnist = np.fromfile(file=fd, dtype=np.uint8).reshape( (60000,784)).astype('float32')

    num_clusters=20

    [clusters, indices]=kmeans(mnist, num_clusters,10)

    for i in xrange(num_clusters):
        plt.subplot(int(np.ceil(num_clusters/5)),5,i+1)
        plt.imshow(clusters[i].reshape(28,28))
    plt.show()
