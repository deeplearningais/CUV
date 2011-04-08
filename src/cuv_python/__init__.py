# vim:sw=4:ts=4:et
import numpy as np
import pyublas
from _cuv_python import *

def _matstr(x,typestr):
    return "%s: (%d,%d)[%2.1f Mb]"%(typestr,x.shape,x.memsize/1024./1024.)

def __cpy(x):
    x2 = x.__class__(x.h,x.w)
    apply_scalar_functor(x2.vec,x.vec,scalar_functor.COPY)
    return x2

def __cpy_dia(x):
    x2 = x.__class__(x)
    apply_scalar_functor(x2.vec,x.vec,scalar_functor.COPY)
    return x2

def __sav_dense(x, file):
    np.save(file.replace(".npy",""),pull(x))

def __shape(x):
    return (x.h,x.w)

def __np(x):
    return pull(x)

def __T(x):
    return transposed_view(x)

def copy(dst,src):
    apply_scalar_functor(dst.vec,src.vec,scalar_functor.COPY)

def __tensor_getitem(x,key):
    if isinstance(key,int):
        return x.get([key])
    return x.get([x for x in key])

def __tensor_setitem(x,key,val):
    if isinstance(key,int):
        return x.set([key],val)
    x.set([x for x in key],val)

def __matrix_getitem__(x,key):
    if isinstance(key,int):
        return x.vec.at(key)
    elif isinstance(key,tuple): # slicing!
        if isinstance(key[0],int) and isinstance(key[1],int):
            return x.at(key[0],key[1]) # single element
        else:
            if isinstance(key[0],slice): # see what first element ist..
                if key[0].start==None:
                    startx=0
                else:
                    startx=key[0].start
                if key[0].stop==None:
                    stopx=x.h
                else:
                    stopx=key[0].stop
            elif isinstance(key[0],int):
                startx=key[0]
                stopx=key[0]+1
            else:
                raise NotImplementedError

            if isinstance(key[1],slice): # see what second element ist..
                if key[1].start==None:
                    starty=0
                else:
                    starty=key[1].start
                if key[1].stop==None:
                    stopy=x.w
                else:
                    stopy=key[1].stop
            elif isinstance(key[1],int):
                starty=key[1]
                stopy=key[1]+1
            else:
                raise NotImplementedError
            return blockview(x,startx,stopx-startx,starty,stopy-starty)
    else:
        print("This slicing is not supported")
        raise NotImplementedError
            

def __matrix_setitem__(x,key,value):
    if isinstance(key,int):
        return x.vec.set(key,value=value)
    elif isinstance(key,tuple): # slicing!
        if isinstance(key[0],int) and isinstance(key[1],int):
            return x.set(key[0],key[1],value=value) # single element
        else:
            view=x.__getitem__(key)
            if view.__class__ != x.__class__:
                raise NotImplementedError("Can only assign matrix of same type")
            copy(view,value)



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
            dense_type.T = property(__T)
            dense_type.has_nan = property(lambda x:has_nan(x))
            dense_type.has_inf = property(lambda x:has_inf(x))
            dense_type.__getitem__=__matrix_getitem__
            dense_type.__setitem__=__matrix_setitem__
            dense_type.__str__=lambda x:(_matstr(x,memory_space+"_matrix_"+memory_layout+value_type))

    dia_type=eval(memory_space+"_dia_matrix_f")
    #dia_type.__str__=lambda x:(_matstr(x,memory_space+"_matrix_"+memory_layout+value_type))

    dia_type.copy = __cpy_dia
    dia_type.shape = property(__shape)
    dia_type.np = property(__np)

for memory_space in ["dev","host"]:
    for value_type in ["float","int","uc","uint"]:
            dense_type=eval(memory_space+"_tensor_"+value_type)

            dense_type.save = __sav_dense
            dense_type.copy = __cpy
            dense_type.T = property(__T)
            dense_type.has_nan = property(lambda x:has_nan(x))
            dense_type.has_inf = property(lambda x:has_inf(x))
            dense_type.__getitem__= __tensor_getitem
            dense_type.__setitem__= __tensor_setitem
            dense_type.__str__=lambda x:(_matstr(x,memory_space+"_tensor_"+value_type))
