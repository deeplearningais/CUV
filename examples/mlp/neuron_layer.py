import pyublas
import cuv_python as cp

class neuron_layer:
    """Neuron layer of MLP consisting of activations and deltas."""

    def __init__(self, dim1, dim2):
        """Constructor

        @param dim1 -- number of neurons in each row layer.
        @param dim2 -- number of neurons in each column of the layer.

        """
        self.activations = cp.get_filled_matrix(dim1, dim2, 0.0)
        self.deltas = cp.get_filled_matrix(dim1, dim2, 0.0)

    def nonlinearity(self, input_):
        """Function applies nonlinerity on every element of the input.

        @param input_ -- input vector/matrix

        """
        cp.apply_scalar_functor(input_, cp.scalar_functor.TANH)

    def d_nonlinearity(self, input_):
			"""Function applies nonlinear derivative on every element of the input.

			@param input_ -- input vector/matrix

			"""
			cp.apply_scalar_functor(input_, cp.scalar_functor.DTANH)

    def _del_(self):
        """Destructor"""
        self.activations.dealloc()
        self.deltas.dealloc()




