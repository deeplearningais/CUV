# vim:ts=4:sw=4:et
import numpy as np
import math
import sys
import pdb
import cuv_python as cp
from nose.tools import *

cp.initCUDA(0)

import unittest

class  testNumpyCompatibility:
    def setUp(self):
        self.shape = [2,3,4]
        pass
    def tearDown(self):
        pass

    def cmp3d_inv(self,t,n):
        """ compares a tensor and a numpy matrix """
        eq_(n.shape[2],t.shape[0])
        eq_(n.shape[1],t.shape[1])
        eq_(n.shape[0],t.shape[2])

        eq_(t.get([0,0,0]), n[0,0,0])
        eq_(t.get([0,0,1]), n[1,0,0])
        eq_(t.get([0,1,0]), n[0,1,0])
        eq_(t.get([1,0,0]), n[0,0,1])

    def cmp3d(self,t,n):
        """ compares a tensor and a numpy matrix """
        eq_(n.shape[0],self.shape[0])
        eq_(t.shape[0],self.shape[0])
        eq_(n.shape[1],self.shape[1])
        eq_(t.shape[1],self.shape[1])
        eq_(n.shape[2],self.shape[2])
        eq_(t.shape[2],self.shape[2])

        eq_(t.get([0,0,0]), n[0,0,0])
        eq_(t.get([0,0,1]), n[0,0,1])
        eq_(t.get([0,1,0]), n[0,1,0])
        eq_(t.get([1,0,0]), n[1,0,0])

    def testNpyToTensor(self):
        """ convert a numpy matrix to a tensor """
        n = np.arange(np.prod(self.shape)).reshape(self.shape).astype("float32")
        t = cp.dev_tensor_float(n)
        self.cmp3d(t,n)

    def testTensorToNpy(self):
        """ convert a tensor to a numpy matrix """
        t = cp.dev_tensor_float(self.shape)
        cp.sequence(t)
        n = t.np
        self.cmp3d(t,n)

    def testNpyToTensorCm(self):
        """ convert a numpy matrix to a tensor (column major)"""
        n = np.arange(np.prod(self.shape)).reshape(self.shape).copy("F").astype("float32")
        t = cp.dev_tensor_float_cm(n)
        self.cmp3d(t,n)

    def testTensorToNpyCm(self):
        """ convert a tensor to a numpy matrix (column major) """
        t = cp.dev_tensor_float_cm(self.shape)
        cp.sequence(t)
        n = t.np
        self.cmp3d(t,n)

    def testNpyToTensorTrans(self):
        """ convert a numpy matrix to a tensor (transposed) """
        n = np.arange(np.prod(self.shape)).reshape(self.shape).copy("F").astype("float32")
        t = cp.dev_tensor_float(n)
        self.cmp3d_inv(t,n)

    def testTensorToNpyTrans(self):
        """ convert a tensor to a numpy matrix (transposed) """
        t = cp.dev_tensor_float_cm(self.shape)
        cp.sequence(t)
        n = t.np
        self.cmp3d(t,n)

    def testNpyToTensorCmTrans(self):
        """ convert a numpy matrix to a tensor (column major, transposed)"""
        n = np.arange(np.prod(self.shape)).reshape(self.shape).astype("float32")
        t = cp.dev_tensor_float_cm(n)
        self.cmp3d_inv(t,n)

    def testTensorToNpyCmTrans(self):
        """ convert a tensor to a numpy matrix (column major, transposed) """
        t = cp.dev_tensor_float(self.shape)
        cp.sequence(t)
        n = t.np
        self.cmp3d(t,n)
