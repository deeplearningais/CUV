import pyublas
import cuv_python as cp

class weight_layer:
    """ weight layer of the MLP represented by a matrix."""

    def __init__(self, source_layer, target_layer):
        """Constructor

        @param source_layer pointer to the previous neuron layer.
        @param target_layer pointer to the next neuron layer.

        """
        self.source=source_layer
        self.target=target_layer
        dim1 = self.target.activations.shape[0]
        dim2 = self.source.activations.shape[0]
        self.weight = cp.get_filled_matrix(dim1, dim2, 0.0)
        cp.fill_rnd_uniform(self.weight)
	self.weight -= 0.5
	self.weight /= 10.0
        self.bias = cp.dev_tensor_float(dim1)
	cp.fill(self.bias, 0)

    def forward(self):
        """Forward pass, calculates the activations of next neuron layer."""
        cp.prod(self.target.activations, self.weight,
                self.source.activations)
        cp.matrix_plus_col(self.target.activations, self.bias)
        self.target.nonlinearity(self.target.activations)

    def backward(self):
        """Backward pass, calculates the deltas of lower layer
           and later updates the weights."""
        cp.prod(self.source.deltas, self.weight, self.target.deltas,
            't',  'n')
        h = cp.dev_tensor_float_cm([self.source.activations.shape[0],
                              self.source.activations.shape[1]])
        cp.apply_scalar_functor(h,  self.source.activations,
                                cp.scalar_functor.COPY)
        self.source.d_nonlinearity(h)
	self.source.deltas *= h
        h.dealloc()
        self.weight_update()

    def weight_update(self, learnrate=0.01, decay=0.0):
        """Updates the weights and the bias
           using source activations and target deltas.

           @param learnrate  how strongly the gradient influences the weights
           @param decay      large values result in a regularization with
                             to the squared weight value"""
        batch_size=self.source.activations.shape[1]
        h = cp.dev_tensor_float_cm([self.weight.shape[0], self.weight.shape[1]])
        cp.prod(h, self.target.deltas, self.source.activations, 'n', 't')
        cp.learn_step_weight_decay(self.weight, h, learnrate/batch_size, decay)
        h.dealloc()
        h = cp.dev_tensor_float(self.target.activations.shape[0])
	cp.fill(h,0)
        cp.reduce_to_col(h, self.target.deltas)
        cp.learn_step_weight_decay(self.bias, h, learnrate/batch_size, decay)
        h.dealloc()

    def _del_(self):
        """Destructor (so we do not need to wait for the garbage collector)"""
        self.weight.dealloc()
        self.bias.dealloc()


