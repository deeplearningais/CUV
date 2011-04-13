import sys
import pyublas
import cProfile
from multi_layer_perceptron import MLP
import cuv_python as cp
from switchtohost import switchtohost
from MNIST_data import MNIST_data

def _tmp(dim1, dim2, value):
    """Function to create a filled matrix.
       This demonstrates how CUV can be extended using python.

    @param dim1 -- number of rows.
    @param dim2 -- number of columns.

    """
    mat = cp.dev_tensor_float_cm([dim1, dim2])
    cp.fill(mat,  value)
    return mat
cp.get_filled_matrix = _tmp


if __name__ == "__main__":
    try:
        if sys.argv[2] == "--host":
            switchtohost()
    except: pass

    try:
        mnist = MNIST_data(sys.argv[1]);
    except:
        print('Usage: %s {path of MNIST dataset} [--host]' % sys.argv[0])
        sys.exit(1)

		# initialize cuv to run on device 0
    cp.initCUDA(0)

		# initialize random number generator with seed 0
    cp.initialize_mersenne_twister_seeds(0)

		# obtain training/test data
    train_data, train_labels = mnist.get_train_data()
    test_data,  test_labels  = mnist.get_test_data()

    # determine layer sizes
    sizes = [train_data.shape[0], 128, train_labels.shape[0]]

    print('Initializing creation of MLP...')
    mlp = MLP(sizes, 96)

    print('Initializing training of  MLP...')
    mlp.train(train_data, train_labels, 100)

    print('Initializing testing of MLP...')
    mlp.test(test_data,test_labels)

    print('done.')
    cp.exitCUDA()

