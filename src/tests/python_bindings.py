import pyublas
import numpy as np
import math
import sys
import pdb
import cuv_python as cp

cp.dev_matrix_cmf.pull = cp.pull
def sigmoid(x):
    return 1.0 / (1.0 +np.exp(-x))
np.sigmoid = sigmoid

import unittest

class TestCuvMatrix(unittest.TestCase):

    def setUp(self):
        self.m = 784
        self.k = 96
        self.n = 512

        self.xa = np.random.standard_normal((self.m,self.k)).astype(np.float32).copy('F')
        self.xb = np.random.random((self.k,self.n)).astype(np.float32).copy('F')
        self.xc = np.random.random((self.m,self.n)).astype(np.float32).copy('F')

        self.a = cp.push(self.xa)
        self.b = cp.push(self.xb)
        self.c = cp.push(self.xc)

    def tearDown(self):
        self.a.dealloc()
        self.b.dealloc()
        self.c.dealloc()
        self.a = None
        self.b = None
        self.c = None

    def testFill(self):
        cp.apply_nullary_functor(self.a,cp.nullary_functor.FILL,0)
        a = self.a.pull()
        for e in a.flat:
            self.assertAlmostEqual(e,0,10)
    def testCreateCopy(self):
        d = cp.dev_matrix_cmf(self.b.h,self.b.w)
        cp.apply_binary_functor(d, self.b,cp.binary_functor.COPY)  # copy b -> xd
        b = cp.pull(self.b) - cp.pull(d)
        for e in b.flat:
            self.assertAlmostEqual( 0, e, 5 )

    def testBinarize(self):
        cp.rnd_binarize(self.a)
        a = cp.pull(self.a)
        for e in a.flat:
            self.assertTrue(e == 1 or e == 0)

    def testRndUniform(self):
        cp.fill_rnd_uniform(self.a)
        a = cp.pull(self.a)
        self.assertTrue( np.abs(a.mean() - 0.5) < 0.3)
        print "Rnd Uniform Stdev: ", a.std()

    def testRnd(self):
        x = self.a
        cp.apply_nullary_functor(x,cp.nullary_functor.FILL,1.3)
        cp.add_rnd_normal(x)
        xx = cp.pull(x)
        self.assertAlmostEqual(np.mean(np.std(xx, 0)), 1, 1)
        self.assertAlmostEqual(np.mean(np.mean(xx, 0)),  1.3, 1)

    def testNorm2(self):
        n = cp.norm2(self.a)**2
        n2 = np.sum(self.a.pull()*self.a.pull())
        self.assertAlmostEqual(n,n2,-1)
    def testScalarFunctor(self):
        cp.apply_scalar_functor(self.a,cp.scalar_functor.EXP)
        cp.apply_scalar_functor(self.b,cp.scalar_functor.SIGM)
        cp.apply_scalar_functor(self.c,cp.scalar_functor.MULT,1.6)
        a = self.a.pull()
        b = self.b.pull()
        c = self.c.pull()
        self.assertAlmostEqual( np.abs(a - np.exp(self.xa)).sum(), 0, 2)
        self.assertAlmostEqual( np.abs(b - np.sigmoid(self.xb)).sum(), 0, 2)
        self.assertAlmostEqual( np.abs(c - self.xc*1.6).sum(), 0, 2)
    def testSubtract(self):
        xa2 = np.random.standard_normal((self.m,self.k)).astype(np.float32).copy('F')
        a2  = cp.push(xa2)
        cp.apply_binary_functor(self.a,a2,cp.binary_functor.SUBTRACT)
        self.assertAlmostEqual( np.abs(self.a.pull() - (self.xa-xa2)).sum(), 0, 3)

    def testProdDev(self):
        cp.prod(self.c,self.a,self.b)
        c = self.c.pull()
        nC = np.dot(self.xa , self.xb)
        self.assertAlmostEqual(np.abs(c-nC).sum(), 0, 0)

    def testProdDev2(self):
        cp.prod(self.c,self.a,self.b, 'n', 'n', 1.3, 1.7)
        c = self.c.pull()
        nC = 1.7 * self.xc + 1.3 * np.dot(self.xa , self.xb)
        self.assertAlmostEqual(np.abs(c-nC).sum(), 0, 0)

    def testProdDev3(self):
        cp.prod(self.b,self.a,self.c,'t','n')
        b = self.b.pull()
        nB = np.dot(self.xa.T , self.xc)
        self.assertAlmostEqual(np.abs(b-nB).sum(), 0, 0)

    def testLearnStepWD(self):
        xa2 = np.random.standard_normal((self.m,self.k)).astype(np.float32).copy('F')
        a2  = cp.push(xa2)
        cp.learn_step_weight_decay(self.a,a2,0.1,0.05)
        #correct = self.xa + 0.1 * (xa2 - 0.05*self.xa)
        correct = (1-0.1*0.05)*self.xa + 0.1 * xa2
        #print self.a.pull(), "\n\n"
        #print correct
        self.assertAlmostEqual( np.abs(self.a.pull() - correct).sum(),0,1)

if __name__ == '__main__':
    dev = 3
    cp.initCUDA(dev);
    cp.initialize_mersenne_twister_seeds(0);
    unittest.main()
    cp.exitCUDA();
