import sys
from multi_layer_perceptron import MLP
import cuv_python as cp
from switchtohost import switchtohost
from MNIST_data import MNIST_data

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

    print('Initializing MLP...')
    mlp = MLP(sizes, 100)

    print('Training MLP...')
    try:
        mlp.train(train_data, train_labels, 100)
    except KeyboardInterrupt:
        pass

    print('Testing MLP...')
    mlp.test(test_data, test_labels)

    print('done.')
    #cp.exitCUDA()

