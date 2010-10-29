import numpy as np
from scipy.io.numpyio import fwrite, fread
import ipdb as pdb
import os
import sys

sys.excepthook = __IPYTHON__.excepthook

path="/home/local/datasets/MNIST"

with open(os.path.join(path,'t10k-images.idx3-ubyte')) as fd:
#with open(os.path.join(path,'train-images.idx3-ubyte')) as fd:
    fread(fd,16,'c')
    #data = np.fromfile(file=fd, dtype=np.uint8).reshape( (60000,784) )
    data = np.fromfile(file=fd, dtype=np.uint8).reshape( (10000,784) )

padded_data=[]
for image in data:
    padded_image=np.zeros((32,32))
    padded_image[2:30,2:30]=image.reshape(28,28)
    padded_data.append(padded_image.flatten())

#np.save(os.path.join(path,'mnist_padded'),np.array(padded_data).T.astype(np.uint8))
np.save(os.path.join(path,'mnist_padded_test'),np.array(padded_data).T.astype(np.uint8))
