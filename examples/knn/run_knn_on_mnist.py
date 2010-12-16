import os
from knn import KNN
import numpy as np



class MNISTPatGen:
  def __init__(self,dir):
      from scipy.io.numpyio import fread
      fd = open(dir+'/train-labels.idx1-ubyte')
      fread(fd,8,'c')
      self.data_labels = np.fromfile(file=fd, dtype=np.uint8).reshape( 60000 )
      fd.close()

      fd = open(dir+'/train-images.idx3-ubyte')
      fread(fd,16,'c')
      self.data = np.fromfile(file=fd, dtype=np.uint8).reshape( (60000,784) )
      fd.close()

      fd = open(dir+'/t10k-images.idx3-ubyte')
      fread(fd,16,'c')
      self.test = np.fromfile(file=fd, dtype=np.uint8).reshape( (10000,784) )
      fd.close()

      fd = open(dir+'/t10k-labels.idx1-ubyte')
      fread(fd,8,'c')
      self.test_labels = np.fromfile(file=fd, dtype=np.uint8).reshape( 10000 )
      fd.close()
  def get_test(self):
      v = self.test.astype('float32').T.copy("F")
      t = self.test_labels
      return v,t
  def get(self):
      v = self.data.astype('float32').T.copy("F")
      t = self.data_labels
      return v,t

pg = MNISTPatGen(os.path.join("/home/local/datasets/MNIST"));

data,data_l  = pg.get()
test, test_l = pg.get_test()

knn = KNN(data.T.copy("F"),data_l,k=1)

def run():
    err_cnt = 0
    off = 5000
    for i in xrange(0,10000,off):
        pred = knn.run(test[:,i:i+off].T.copy("F"))
        err_cnt += (pred!=test_l[i:i+off]).sum()
    return err_cnt
print "Errors: ", run()

#import timeit
#print timeit.Timer("run()", "from __main__ import run, data, data_l, test, test_l, knn").timeit(number=2)/2.


