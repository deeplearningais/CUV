import cuv_python as cp
import pyublas
from neuron_layer import neuron_layer
from weight_layer import weight_layer

class MLP:
    """
		A Multi-Layer Perceptron
		"""
    def __init__(self, neurons, batch_size):
        """Constructor

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
        """Function to train the network

        @param input_matrix -- matrix consisting of input data
           to the network.
        @param teacher_matrix -- matrix consisting of labels
           of input data.
        @param number_of_epochs -- number of rounds the network
           is to be trained.

        """
        number_of_pictures = input_matrix.shape[-1]
        squared_errors = cp.dev_matrix_cmf(self.neuron_layer[-1].deltas.h,
                                           self.neuron_layer[-1].deltas.w)
        for r in xrange(number_of_epochs):
            print "Epoch ", r+1, "/", number_of_epochs
            mse = 0
            for batch in xrange(number_of_pictures/self.batch_size):
                index_begin = self.batch_size * batch
                index_end   = self.batch_size + index_begin

                # Push input and teacher to GPU memory
                self.neuron_layer[0].activations = cp.push(
                    input_matrix[:,index_begin:index_end].astype('float32').copy('F'))
                teachbatch = cp.push(
                    teacher_matrix[:,index_begin:index_end].astype('float32').copy('F'))

                # Forward-Pass
                for i in xrange(self.number_of_layers):
                    self.weight_layer[i].forward()

                # calculate error at output layer
                cp.apply_binary_functor(self.neuron_layer[-1].deltas,
                    teachbatch, cp.binary_functor.COPY)
                cp.apply_binary_functor(self.neuron_layer[-1].deltas,
                    self.neuron_layer[-1].activations,
                        cp.binary_functor.SUBTRACT)
                cp.apply_binary_functor(squared_errors, self.neuron_layer[-1].deltas,
                    cp.binary_functor.COPY)
                cp.apply_scalar_functor(squared_errors, cp.scalar_functor.SQUARE)
                mse += cp.sum(squared_errors)


                # Backward-Pass
                for i in xrange(self.number_of_layers):
                    self.weight_layer[self.number_of_layers-i-1].backward()

                # Don't wait for garbage collector
                teachbatch.dealloc()
                self.neuron_layer[0].activations.dealloc()

            print "MSE: ", (mse/number_of_pictures)
        squared_errors.dealloc()

    def test(self, input_matrix, teacher_matrix):
        """Function to test the network

        @param input_matrix -- matrix consisting of input
           data to the network.
        @param teacher_matrix -- matrix consisting of labels
           of input data .

        """
        number_of_pictures = input_matrix.shape[-1]
        mse = 0
        squared_errors = cp.dev_matrix_cmf(self.neuron_layer[-1].deltas.h,
            self.neuron_layer[-1].deltas.w)
        for batch in xrange(number_of_pictures/self.batch_size):
            index_begin = self.batch_size * batch
            index_end = index_begin + self.batch_size
            self.neuron_layer[0].activations = cp.push( input_matrix[:,
                index_begin:index_end].astype('float32').copy('F'))
            teachbatch = cp.push(teacher_matrix[:,
                index_begin:index_end].astype('float32').copy('F'))
            for i in xrange(self.number_of_layers):
                self.weight_layer[i].forward()
            cp.apply_binary_functor(squared_errors, self.neuron_layer[-1].deltas,
                cp.binary_functor.COPY)
            cp.apply_scalar_functor(squared_errors, cp.scalar_functor.SQUARE)
            mse += cp.sum(squared_errors)
            teachbatch.dealloc()
        print "MSE: ", (mse/number_of_pictures)
        squared_errors.dealloc()



