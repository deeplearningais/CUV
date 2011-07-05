import cuv_python as cp
import numpy as np
from neuron_layer import neuron_layer
from weight_layer import weight_layer


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
        self.number_of_layers = len(neurons) - 1
        self.batch_size = batch_size
        self.neuron_layer = []
        self.weight_layer = []
        for i in xrange(self.number_of_layers+1):
            dim1 = neurons[i]
            self.neuron_layer.append(neuron_layer(dim1,
                self.batch_size ))
        for i in xrange(self.number_of_layers):
            self.weight_layer.append(weight_layer(self.neuron_layer[i],
                self.neuron_layer[i+1]))

    def train(self, input_matrix, teacher_matrix, number_of_epochs):
        """
        Function to train the network

        @param input_matrix -- matrix consisting of input data
           to the network.
        @param teacher_matrix -- matrix consisting of labels
           of input data.
        @param number_of_epochs -- number of rounds the network
           is to be trained.

        """
        number_of_pictures = input_matrix.shape[-1]
        squared_errors = cp.dev_tensor_float_cm([self.neuron_layer[-1].deltas.shape[0],
        self.neuron_layer[-1].deltas.shape[1]])
        for r in xrange(number_of_epochs):
            print "Epoch ", r+1, "/", number_of_epochs
            mse = 0.0
            ce  = 0.0
            for batch in xrange(number_of_pictures/self.batch_size):
                index_begin = self.batch_size * batch
                index_end   = self.batch_size + index_begin

                # Push input and teacher to GPU memory
                self.neuron_layer[0].activations = cp.dev_tensor_float_cm(
                    input_matrix[:,index_begin:index_end].astype('float32').copy('F'))
                teachbatch_host = teacher_matrix[:,index_begin:index_end]
                teachbatch = cp.dev_tensor_float_cm(teachbatch_host.astype('float32').copy('F'))

                # Forward-Pass
                for i in xrange(self.number_of_layers):
                    self.weight_layer[i].forward()

                # calculate error at output layer
                cp.copy(self.neuron_layer[-1].deltas, teachbatch)
                cp.apply_binary_functor(self.neuron_layer[-1].deltas,
                    self.neuron_layer[-1].activations,
                        cp.binary_functor.SUBTRACT)
                cp.copy(squared_errors, self.neuron_layer[-1].deltas)
                cp.apply_scalar_functor(squared_errors, cp.scalar_functor.SQUARE)
                mse += cp.sum(squared_errors)
                ce  += float(np.sum(np.argmax(teachbatch_host,axis=0)
			!=          np.argmax(self.neuron_layer[-1].activations.np,axis=0)))


                # Backward-Pass
                for i in xrange(self.number_of_layers):
                    self.weight_layer[self.number_of_layers-i-1].backward()

                # Don't wait for garbage collector
                teachbatch.dealloc()
                self.neuron_layer[0].activations.dealloc()

            print "MSE: ",     (mse / number_of_pictures)
            print "ClassErr: ", (ce / number_of_pictures)
        squared_errors.dealloc()

    def test(self, input_matrix, teacher_matrix):
        """
        Function to test the network

        @param input_matrix -- matrix consisting of input
           data to the network.
        @param teacher_matrix -- matrix consisting of labels
           of input data .

        """
        number_of_pictures = input_matrix.shape[-1]
        mse = 0.0
        ce  = 0.0
        squared_errors = cp.dev_tensor_float_cm([self.neuron_layer[-1].deltas.shape[0],
            self.neuron_layer[-1].deltas.shape[1]])
        for batch in xrange(number_of_pictures/self.batch_size):
            index_begin = self.batch_size * batch
            index_end = index_begin + self.batch_size
            self.neuron_layer[0].activations = cp.dev_tensor_float_cm( input_matrix[:,
                index_begin:index_end].astype('float32').copy('F'))
            teachbatch = cp.dev_tensor_float_cm(teacher_matrix[:,
                index_begin:index_end].astype('float32').copy('F'))
            for i in xrange(self.number_of_layers):
                self.weight_layer[i].forward()
            cp.copy(squared_errors, self.neuron_layer[-1].deltas)
            cp.apply_scalar_functor(squared_errors, cp.scalar_functor.SQUARE)
            mse += cp.sum(squared_errors)
            ce  += float(np.sum(np.argmax(teachbatch.np,axis=0)
                    !=          np.argmax(self.neuron_layer[-1].activations.np,axis=0)))
            teachbatch.dealloc()
        print "MSE: ", (mse/number_of_pictures)
        print "ClassErr: ", (ce/number_of_pictures)
        squared_errors.dealloc()
