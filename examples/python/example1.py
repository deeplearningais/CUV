import cuv_python as cp
import numpy as np

h = np.zeros((1,256))                                   # create numpy matrix
d = cp.dev_tensor_float(h)                              # constructs by copying numpy_array

h2 = np.zeros((1,256)).copy("F")                        # create numpy matrix
d2 = cp.dev_tensor_float_cm(h2)                         # creates dev_tensor_float_cm (column-major float) object

cp.fill(d,1)                                            # terse form
cp.apply_nullary_functor(d,cp.nullary_functor.FILL,1)   # verbose form

h = d.np                                                # pull and convert to numpy
assert(np.sum(h) == 256)
assert(cp.sum(d) == 256)
d.dealloc()                                             # explicitly deallocate memory (optional)
