import cuv_python as cp
import pyublas
import numpy as np

class KNN:
    def __init__(self, data, data_l, k):
        self.k      = k
        self.data   = cp.push(data)
        self.data_l = data_l
        self.dsq    = cp.dev_matrix_cmf(self.data.h,1)
        cp.reduce_to_col(self.dsq.vec,self.data,cp.reduce_functor.ADD_SQUARED)
    def get_distance_matrix(self, test):
        t   = cp.push(test)
        assert t.w == self.data.w
        tsq = cp.dev_matrix_cmf(t.h, 1)
        cp.reduce_to_col(tsq.vec,t,cp.reduce_functor.ADD_SQUARED)
        p   = cp.dev_matrix_cmf(self.data.h, t.h)
        cp.prod(p, self.data, t, 'n','t',-2, 0)
        cp.matrix_plus_col(p,self.dsq.vec)
        cp.matrix_plus_row(p,tsq.vec)
        return p
    def run(self,test):
        p = self.get_distance_matrix(test)
        p *= -1.                # no argmin supported yet
        idx = cp.dev_matrix_cmi(test.shape[0],1)
        cp.argmax_to_row(idx.vec, p)
        hidx  = idx.np.reshape(idx.h)
        return self.data_l.reshape(self.data.h)[hidx]
