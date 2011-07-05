import numpy as np


class MNIST_data:
    """Input data for training and testing the MLP from MNIST dataset"""

    def __init__(self, dir):
        """
        Constructor

        @param dir -- path of MNIST dataset

        """
        with open(dir + '/train-labels.idx1-ubyte') as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=8)
            self.data_labels = np.fromfile(file=fd, dtype=np.uint8).reshape(60000)

        with open(dir + '/train-images.idx3-ubyte') as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=16)
            self.data = np.fromfile(file=fd, dtype=np.uint8).reshape((60000, 784))

        with open(dir + '/t10k-images.idx3-ubyte') as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=16)
            self.test = np.fromfile(file=fd, dtype=np.uint8).reshape((10000, 784))

        with open(dir + '/t10k-labels.idx1-ubyte') as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=8)
            self.test_labels = np.fromfile(file=fd, dtype=np.uint8).reshape(10000)

    def get_test_data(self):
        """Function returning testing data and label matrices"""
        data = (((self.test.astype('float32') / 255.0) - 0.5) * 2).T
        #creating a label matrix suitable for MLP
        label = -0.90 * np.ones((10, 10000)).astype('float32')
        tmp = np.vstack((self.test_labels.T, np.arange(10000).T))
        #assigning correct label
        label[tmp[0], tmp[1]] = 0.90
        data_ = data.copy('F')
        label_ = label.astype('float32').copy('F')
        return data_, label_

    def get_train_data(self):
        """Function returning training data and label matrices"""
        data = (((self.data.astype('float32') / 255.0) - 0.5) * 2).T

        #creating a matrix of labels suitable for MLP
        label = -0.90 * np.ones((10, 60000), np.float32)
        tmp = np.vstack((self.data_labels.T, np.arange(60000).T))

        #assigning positive value to row in each clolumn based on corresponding data label
        label[tmp[0], tmp[1]] = 0.90
        data_ = data.copy('F')
        label_ = label.astype('float32').copy('F')
        return data_, label_
