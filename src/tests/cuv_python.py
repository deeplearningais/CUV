import pyublas
import numpy
import math
import sys
import pdb
sys.path.append('../../build/debug/python_bindings')
import cuv_python

import unittest

class TestCuvMatrix(unittest.TestCase):

    def setUp(self):

        self.xa = numpy.random.standard_normal((5,6)).astype(numpy.float32).copy('F')
        self.xb = numpy.random.random((5,6)).astype(numpy.float32).copy('F')

        self.a = cuv_python.create_mat_from_numpy_view("matrix_a", self.xa)
        self.b = cuv_python.create_mat_from_numpy_view("matrix_b", self.xb)

        self.xc = cuv_python.create_numpy_from_mat_copy(self.a)
        self.c  = cuv_python.create_mat_from_numpy_view("matrix_c",self.xc)

        self.xd = cuv_python.create_numpy_from_mat_copy(self.a) # copy a -> xd
        self.d  = cuv_python.create_mat_from_numpy_view("matrix_d",self.xd)
        cuv_python.copy_numpy_to_host_matrix(self.d, self.xb)  # copy b -> xd

        self.a.push()
        self.b.push()

    def tearDown(self):
        self.a.deallocDevice()
        self.b.deallocDevice()
        self.a = None
        self.b = None

    def testCreateCopy(self):
        self.a[0,0]=5
        self.c[0,0]=6  # c was created by _copying_ a
        self.assertEqual(self.a[0,0], 5)
        self.assertEqual(self.c[0,0], 6)

    def testBinarize(self):
        cuv_python.binarize_probs(self.a)
        self.a.pull()
        for e in self.xa.flat:
            self.assertTrue(e == 1 or e == 0)
    def testRndUniform(self):
        cuv_python.init_rnd_uniform(self.a)
        self.a.pull()
        print self.xa

    def testRnd(self):
        xx = numpy.zeros((2000,10), dtype='float32', order='F')
        x = cuv_python.create_mat_from_numpy_view("rng_test", xx)
        x.allocDevice()
        x.setDevToVal(1.3)
        cuv_python.add_gaussian_noise(x, 2.5)
        x.pull()
        self.assertAlmostEqual(numpy.mean(numpy.std(xx, 0)), 2.5, 1)
        self.assertAlmostEqual(numpy.mean(numpy.mean(xx, 0)),  1.3, 1)
        x.deallocDevice()

        
    def testScalarFunctor(self):
        self.a[0,0] = 3
        self.a[0,1] = 5
        self.a.push()
        cuv_python.apply_scalar_functor_host(cuv_python.ScalarFunctor.EXP, self.a)
        cuv_python.apply_scalar_functor_device(cuv_python.ScalarFunctor.EXP, self.a)
        self.assertAlmostEqual(self.a[0,0], math.exp(3), 4)
        self.assertAlmostEqual(self.a[0,1], math.exp(5), 4)
        self.a.pull()
        self.assertAlmostEqual(self.a[0,0], math.exp(3), 4)
        self.assertAlmostEqual(self.a[0,1], math.exp(5), 4)

    def testProdDev(self):
        xC = numpy.zeros((5,5), dtype='float32', order='F')
        C = cuv_python.create_mat_from_numpy_view("prod_host_res",xC)
        C.dMultiply(self.a,self.b, 'n','t')
        C.pull()
        nC = numpy.dot(self.xa , self.xb.T)
        self.assertAlmostEqual((xC-nC).sum(), 0, 5)

if __name__ == '__main__':
    dev = 3
    cuv_python.cuv_init(dev);
    cuv_python.init_RNG( 1, "rnd_multipliers_32bit.txt")
    unittest.main()
    cuv_python.cuv_exit(dev);
