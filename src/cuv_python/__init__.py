import numpy as np
from _cuv_python import *

def __cpy(x):
    x2 = x.__class__(x.h,x.w)
    apply_binary_functor(x2.vec,x.vec,binary_functor.COPY)
    return x2

def __cpy_dia(x):
    x2 = x.__class__(x)
    apply_binary_functor(x2.vec,x.vec,binary_functor.COPY)
    return x2

def __sav_dense(x, file):
    np.save(file.replace(".npy",""),pull(x))

def __shape(x):
    return (x.h,x.w)

def __np(x):
    return pull(x)

def __getitem__(x,key):
    if key.__class__==int:
        return x.vec.at(key)
    else:
        return x.at(*key)

def __setitem__(x,key,value):
    x.set(*key,value=value)

def copy(dst,src):
    apply_binary_functor(dst.vec,src.vec,binary_functor.COPY)

# Combine strings to form all exported combinations of types
# For all types add convenience functions

for memory_space in ["dev","host"]:
    for value_type in ["f","sc","uc","i", "ui"]:
        for memory_layout in ["rm","cm"]:
            dense_type=eval(memory_space+"_matrix_"+memory_layout+value_type)

            dense_type.save = __sav_dense
            dense_type.copy = __cpy
            dense_type.shape = property(__shape)
            dense_type.np = property(__np)
            dense_type.has_nan = property(lambda x:has_nan(x))
            dense_type.has_inf = property(lambda x:has_inf(x))
            dense_type.__getitem__=__getitem__
            dense_type.__setitem__=__setitem__

    dia_type=eval(memory_space+"_dia_matrix_f")

    dia_type.copy = __cpy_dia
    dia_type.shape = property(__shape)
    dia_type.np = property(__np)
