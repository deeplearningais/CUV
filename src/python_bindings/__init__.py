import numpy as np
from _cuv_python import *

def __cpy(x):
    x2 = x.__class__(x.h,x.w)
    cp.apply_binary_functor(x2.vec,x.vec,cp.binary_functor.COPY)
    return x2

def __sav_dense(x, file):
    np.save(file.replace(".npy",""),cp.pull(x))

def __shape(x):
    return (x.h,x.w)

def __np(x):
    return cp.pull(x)

dev_matrix_cmf.save = __sav_dense
dev_matrix_cmf.copy = __cpy
dev_matrix_cmf.shape = property(__shape)
dev_matrix_cmf.np = property(__np)
dev_matrix_cmf.has_nan = property(lambda x:cp.has_nan(x))
dev_matrix_cmf.has_inf = property(lambda x:cp.has_inf(x))
dev_dia_matrix_f.shape = property(__shape)
dev_dia_matrix_f.np = property(__np)
