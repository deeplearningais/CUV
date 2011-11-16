import cuv_python as cp
import numpy as np
from neuron_layer import neuron_layer
from weight_layer import weight_layer

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

class MLP:
    """
    A Multi-Layer Perceptron
    """
    def __init__(self, neurons, batch_size):
        """
        Constructor

        @param neurons -- array of sizes of layers.
        @param batch_size -- size of batch being used for training.

        """
        self.n_layers = len(neurons) - 1
        self.batch_size = batch_size
        self.neuron_layers = []
        self.weight_layers = []
        print("Training MLP with %d hidden layer(s)." % (self.n_layers - 1))
        for i in xrange(self.n_layers + 1):
            dim1 = neurons[i]
            self.neuron_layers.append(neuron_layer(dim1,
                self.batch_size))
        for i in xrange(self.n_layers):
            self.weight_layers.append(weight_layer(self.neuron_layers[i],
                self.neuron_layers[i + 1]))

    def fit(self, input_matrix, teacher_matrix, n_epochs=100, learnrate = 0.10):
        """
        Function to train the network

        @param input_matrix -- matrix consisting of input data
           to the network.
        @param teacher_matrix -- matrix consisting of labels
           of input data.
        @param n_epochs -- number of epochs the network
           is to be trained.

        """
        number_of_pictures = input_matrix.shape[-1]
        squared_errors = cp.dev_tensor_float_cm(self.neuron_layers[-1].deltas.shape)
        for r in xrange(n_epochs):
            print "Epoch ", r + 1, "/", n_epochs
            mse = 0.0
            ce = 0.0
            for batch in xrange(number_of_pictures / self.batch_size):
                index_begin = self.batch_size * batch
                index_end = self.batch_size + index_begin

                # Push input and teacher to GPU memory
                # .copy("F") is needed since memory is non-contiguous
                self.neuron_layers[0].activations = cp.dev_tensor_float_cm(
                    input_matrix[:, index_begin:index_end].copy('F'))
                teacher_batch_host = teacher_matrix[:, index_begin:index_end]
                teacher_batch = cp.dev_tensor_float_cm(teacher_batch_host.copy('F'))

                # Forward-Pass
                for i in xrange(self.n_layers):
                    self.weight_layers[i].forward()

                # calculate error at output layer
                cp.copy(self.neuron_layers[-1].deltas, teacher_batch)
                self.neuron_layers[-1].deltas -= self.neuron_layers[-1].activations
                cp.copy(squared_errors, self.neuron_layers[-1].deltas)
                cp.apply_scalar_functor(squared_errors, cp.scalar_functor.SQUARE)
                mse += cp.sum(squared_errors)
                ce += float(np.sum(np.argmax(teacher_batch_host, axis=0)
                        != np.argmax(self.neuron_layers[-1].activations.np, axis=0)))

                # Backward-Pass
                for i in xrange(self.n_layers):
                    self.weight_layers[self.n_layers - i - 1].backward(learnrate, decay = .01)

                # Don't wait for garbage collector
                teacher_batch.dealloc()
                self.neuron_layers[0].activations.dealloc()

            print "MSE: ",     (mse / number_of_pictures)
            print "Classification Error Training: ", (ce / number_of_pictures)
        squared_errors.dealloc()

    def predict(self, input_matrix):
        """
        Predict label on unseen data

        @param input_matrix -- matrix consisting of input
           data to the network.

        """
        number_of_pictures = input_matrix.shape[-1]
        predictions = []
        for batch in xrange(number_of_pictures / self.batch_size):
            index_begin = self.batch_size * batch
            index_end = index_begin + self.batch_size
            self.neuron_layers[0].activations = cp.dev_tensor_float_cm(input_matrix[:,
                index_begin:index_end].copy('F'))
            for i in xrange(self.n_layers):
                self.weight_layers[i].forward()
            prediction_batch = np.argmax(self.neuron_layers[-1].activations.np, axis=0)
            predictions.append(prediction_batch)
        return np.hstack(predictions)
