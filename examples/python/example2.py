import cuv_python as cp
import numpy as np

C = cp.dev_tensor_float_cm([2048,2048])   # column major tensor
A = cp.dev_tensor_float_cm([2048,2048])
B = cp.dev_tensor_float_cm([2048,2048])
cp.fill(C,0)                       # fill with some defined values, not really necessary here
cp.sequence(A)
cp.sequence(B)
cp.apply_binary_functor(B,A,cp.binary_functor.MULT) # elementwise multiplication
B *= A                                              # operators also work (elementwise)
cp.prod(C,A,B,'n','t')                              # matrix multiplication
