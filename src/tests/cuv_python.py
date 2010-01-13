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

    def testView(self):
        a2 = cuv_python.matrix_view("view_a", self.a, 0,5,1,5)
        a2.pull()
        x1 = a2[0,0]
        x2 = a2[2,2]
        self.a.pull()
        y1 = self.a[0,1]
        y2 = self.a[2,3]
        self.assertEqual(x1,y1)
        self.assertEqual(x2,y2)
        self.assertEqual(a2.h(), 5)
        self.assertEqual(a2.w(), 4)
        a2.deallocHost()

    def testSetRow(self):
        self.a.setDevToVal(0)
        self.a.dSetRow(3,7.8)
        self.a.pull()
        v = numpy.mean(self.xa,1)
        self.assertAlmostEqual(v[0],0)
        self.assertAlmostEqual(v[1],0)
        self.assertAlmostEqual(v[2],0)
        self.assertAlmostEqual(v[3],7.8,5)
        self.assertAlmostEqual(v[4],0)
        

    def testSumCol(self):
        a = cuv_python.FloatMatrix("a", 17,16,False)
        a.setDevToVal(5.7)
        a.dSet(4,5,2)
        a.dSet(5,3,2)
        a.dSet(16,8,2)
        v = cuv_python.FloatMatrix("v", 17,1,False)
        v.dSumColumns(a)
        v.pull()
        try1 = cuv_python.create_numpy_from_mat_copy(v)
        cuv_python.fast_col_sum(v,a)
        v.pull()
        try2 = cuv_python.create_numpy_from_mat_copy(v)
        self.assertAlmostEqual(numpy.sum(try1-try2),0,3)
        a.dealloc()
        v.dealloc()

    def testSumRow(self):
        a_ =(numpy.random.standard_normal((128,256))/10).astype('float32').copy('F')
        a = cuv_python.create_mat_from_numpy_copy("a",a_)
        a.push()
        v = cuv_python.FloatMatrix("v", 1,256,False)
        cuv_python.fast_row_sum(v,a)
        v.pull()
        try2 = cuv_python.create_numpy_from_mat_copy(v)
        self.assertAlmostEqual(numpy.sum(numpy.sum(a_,axis=0)-try2),0,3)
        a.dealloc()
        v.dealloc()

    def testMultiplyColP(self):
        self.a.setDevToVal(1)
        v = cuv_python.FloatMatrix("v", 1,5,False)
        for i in xrange(0,5):
            v.dSet(0,i,i)
        self.a.dMultiplyColP(v)
        self.a.pull()
        tst = numpy.sum(self.xa,1)
        self.assertAlmostEqual(tst[0], 6*0)
        self.assertAlmostEqual(tst[1], 6*1)
        self.assertAlmostEqual(tst[2], 6*2)
        self.assertAlmostEqual(tst[3], 6*3)
        self.assertAlmostEqual(tst[4], 6*4)

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
    
    def testProdHost(self):
        xC = numpy.zeros((5,5), dtype='float32', order='F')
        C = cuv_python.create_mat_from_numpy_view("prod_host_res",xC)
        cuv_python.prod_host(C,self.a,'n',self.b, 't',False)
        nC = numpy.dot(self.xa , self.xb.T)
        self.assertAlmostEqual((xC-nC).sum(), 0, 5)

    def testProdDev(self):
        xC = numpy.zeros((5,5), dtype='float32', order='F')
        C = cuv_python.create_mat_from_numpy_view("prod_host_res",xC)
        C.dMultiply(self.a,self.b, 'n','t')
        C.pull()
        nC = numpy.dot(self.xa , self.xb.T)
        self.assertAlmostEqual((xC-nC).sum(), 0, 5)
    
    def testAdd(self):
        self.a.setDevToVal(3.7)
        self.b.setDevToVal(3.7)
        self.a.dAdd(self.b)
        self.a.pull()
        for e in self.xa.flat:
                self.assertAlmostEqual(e, 7.4, 5)
    #def testAddLarge(self):
        #x = cuv_python.FloatMatrix("x", 580,2894,False)
        #x.setDevToVal(0)
        #x.dAdd(3)
        #x = cuv_python.create_numpy_from_mat_copy(x)
        #for e in x.flat:
                #self.assertAlmostEqual(e, 7.4, 5)

    def testMultScalarAndAdd(self):
        self.a.setDevToVal(3.7)
        self.a.dMultiplyScalarAndAdd(3, 8)
        self.a.pull()
        for e in self.xa.flat:
                self.assertAlmostEqual(e, 3.7*3+8, 5)

    def testSubtract1(self):
        self.a.setDevToVal(7.4)
        self.b.setDevToVal(3.7)

        self.a.dSubtract(self.b)
        self.a.pull()
        for e in self.xa.flat:
                self.assertAlmostEqual(e, 3.7, 5)

    def testSubtract2(self):
        self.a.setDevToVal(7.4)
        self.b.setDevToVal(3.7)

        self.a.dSubtract(self.b,self.b)
        self.a.pull()
        for e in self.xa.flat:
                self.assertAlmostEqual(e, 0.0, 5)
    
    def testInverse(self):      
        cuv_python.apply_scalar_functor_device(cuv_python.ScalarFunctor.INV,self.a)     
        self.a.pull()
        for i in xrange(self.a.h()): 
            for j in xrange(self.a.w()):
                  self.assertAlmostEqual(self.a.at(i,j),1/(self.c.at(i,j) +
                                                           0.00000001),5)
    
    def testSublin(self):
        cuv_python.apply_scalar_functor_device(cuv_python.ScalarFunctor.SUBLIN,self.a)
        self.a.pull()
        for i in xrange(self.a.h()): 
            for j in xrange(self.a.w()):
                self.assertAlmostEqual(self.a.at(i,j),1- self.c.at(i,j),5)

    def testEnerg(self):
        cuv_python.apply_scalar_functor_device(cuv_python.ScalarFunctor.ENERG,self.a)
        self.a.pull()
        for i in xrange(self.a.h()) :
            for j in xrange(self.a.w()) :
                self.assertAlmostEqual(self.a.at(i,j),math.log(1+self.c.at(i,j)),5)
    
    def testBitflip(self):
        cuv_python.bitflip_row(self.a,2)
        self.a.pull()
        for i in xrange(self.a.h()) :
            for j in xrange(self.a.w()) :
                self.assertAlmostEqual(self.a.at(i,j),(self.c.at(i,j) ,
                                       1-self.c.at(i,j))[i==2],5)
    def testSetDiag(self):
        self.a.dSetDiag(0.2)
        self.a.pull()
        for i in xrange(self.a.h()) :
            for j in xrange(self.a.w()) :
                if i==j: self.assertAlmostEqual(self.a.at(i,j), 0.2, 7)
                else:    self.assertAlmostEqual(self.a.at(i,j), self.c.at(i,j), 7)



    def testMultRowP(self):
        row = cuv_python.FloatMatrix("row",1,self.a.w(),True)
        row.setHostToVal(2)
        row.push()
        row.dSet(1,1,5)
        row.pull()
        row.prnt(False)
        self.a.prnt(False)
        self.a.dMultiplyRowP(row)
        self.a.pull()
        self.a.prnt(False)
        for i in xrange(self.a.h()) :
            for j in xrange(self.a.w()) :
                self.assertAlmostEqual(self.a.at(i,j),self.c.at(i,j)*row.at(0,j))
    
    def testDTranspose(self):
        self.a.push()
        self.a.dTranspose()
        self.a.pull()
        self.c.transpose()
        for i in xrange(self.a.h()) :
            for j in xrange(self.a.w()) :
                self.assertAlmostEqual(self.a.at(i,j),self.c.at(i,j))

if __name__ == '__main__':
    dev = 3
    cuv_python.cuv_init(dev);
    cuv_python.init_RNG( 1, "rnd_multipliers_32bit.txt")
    unittest.main()
    cuv_python.cuv_exit(dev);
