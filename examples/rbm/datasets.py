#from scipy.cluster.vq import whiten
#import mdp
import base as pyrbm
import pdb as pdb
import re,os,sys
import numpy as np
from glob import glob
import Image


class DataSet:
    def __init__(self):
        pass
    def make_cmf(self):
        """ convert to column major float32 """
        self.ensure_float32()
        self.ensure_cm()

    def normalize(self):
        """ normalize between 0 and 1 """
        vmin = self.data.min()
        vmax = self.data.max()
        self.data -= vmin
        self.data /= float(vmax-vmin)
        if "test_data" in self.__dict__:
            self.test_data -= vmin
            self.test_data /= float(vmax-vmin)

    def ensure_cm(self):
        """ makes sure that everything is in column-major format """
        map( self._ensure_cm, ["data", "test_data", "teacher", "test_teacher"])

    def _ensure_cm(self,var):
        """ makes sure that self.var is in column-major format if it exists """
        if var in self.__dict__:
            if not getattr(self,var).flags.fortran: setattr(self,var, getattr(self,var).copy('F'))

    def ensure_float32(self):
        """ makes sure that everything is in float32 format """
        map( self._ensure_float32, ["data", "test_data", "teacher", "test_teacher"])

    def _ensure_float32(self,var):
        """ makes sure that self.var is in float32 format if it exists """
        if var in self.__dict__:
            if getattr(self,var).dtype != np.dtype('float32'): setattr(self,var, getattr(self,var).astype('float32'))

    def logtransform(self):
        """ adds one to (test) data and applies log """
        self.ensure_float32()
        self.data += 1
        np.log(self.data,self.data)
        if "test_data" in self.__dict__:
            self.test_data += 1
            np.log(self.test_data, self.test_data)

    def shuffle(self):
        """ changes the order of data and the teacher (if existent).
            assumes that entries in data are stored in columns
        """
        idx = np.arange(self.data.shape[1])
        np.random.shuffle(idx)
        self.data    = self.data[:,idx]
        if "teacher" in self.__dict__ and self.teacher!=None:
            self.teacher = self.teacher[:,idx]

    def prepare_teacher(self,labels,num_labels,cfg):
        """ prepares a teacher-matrix (one-out-of-n) from a set of labels """
        #v = [[0.0,1.0],[0.,1.]][bool(cfg.finetune_softmax)] I am not wise enough to understand this line yet - and not the (?:) part, the 0.0 vs 0. part I mean.
        v = [0.0,1.0]
        teacher = v[0]*np.ones((num_labels,len(labels)))
        tup = np.vstack(( labels.T, np.arange(len(labels)).T))
        teacher[tup[0],tup[1]] = v[1] # correct label 
        teacher=teacher.astype('float32').copy("F")
        return teacher

    def subtract_variable_mean(self):
        """ subtract mean from each variable """
        self.mean = self.data.mean(axis=1)
        self.data -= self.mean[:,np.newaxis]
        if "test_data" in self.__dict__:
            self.test_data -= self.mean[:,np.newaxis]

    def variable_unit_variance(self):
        """ divide each variable by its variance """
        self.std = np.std(self.data,axis=1)
        self.std[self.std<0.01] = 0.01
        self.data /= self.std[:,np.newaxis]
        if "test_data" in self.__dict__:
            self.test_data /= self.std[:,np.newaxis]

class MNISTPadded(DataSet):
    def __init__(self,cfg, path):
        cfg.px=cfg.py=32
        cfg.num_classes =10
        cfg.test_batchsize=16*25
        if cfg.batchsize==-1:
            cfg.batchsize=16*25

        # load training data
        self.data=np.load(os.path.join(path,"mnist_padded.npy")).astype(np.uint8)
        # load test data
        self.test_data=np.load(os.path.join(path,"mnist_padded_test.npy")).astype(np.uint8)

        # load training labels
        with open(os.path.join(path,'train-labels.idx1-ubyte')) as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=8)
            data_labels = np.fromfile(file=fd, dtype=np.uint8).reshape( (60000,1) )
        self.teacher = self.prepare_teacher(data_labels,10,cfg)
        # load test labels
        with open(os.path.join(path,'t10k-labels.idx1-ubyte')) as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=8)
            test_labels = np.fromfile(file=fd, dtype=np.uint8).reshape( (10000,1) )
        self.test_teacher = self.prepare_teacher(test_labels,10,cfg)


class ShifterData(DataSet):
    def __init__(self,cfg, path):
        cfg.px=19
        cfg.py=1
        cfg.maps_bottom=1
        data=[] 
        self.teacher=None
        if cfg.batchsize==-1:
            cfg.batchsize=768

        # load training data
        with open(os.path.join(path,'Shifter.txt')) as fd:
            for line in fd.readlines():
                data.append(np.array([int(x) for x in line[:-1]]))
        self.data=(np.vstack(data)).astype(np.float32).T

class BarsAndStripesData(DataSet):
    def __init__(self,cfg, path):
        cfg.px=16
        cfg.py=1
        cfg.maps_bottom=1
        data=[] 
        self.teacher=None
        # load training data
        with open(os.path.join(path,'BarsAndStripes.txt')) as fd:
            for line in fd.readlines():
                data.append(np.array([int(x) for x in line[:-1]]))
        if cfg.batchsize==-1:
            cfg.batchsize=32
        self.data=(np.vstack(data)).astype(np.float32).T

class MNISTData(DataSet):
    def __init__(self,cfg, path):
        cfg.px=cfg.py=28
        cfg.num_classes =10
        cfg.test_batchsize=16*25
        cfg.maps_bottom=1

        if cfg.batchsize==-1:
            cfg.batchsize=16*25
        # load training data
        with open(os.path.join(path,'train-images.idx3-ubyte')) as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=16)
            self.data = np.fromfile(file=fd, dtype=np.uint8).reshape( (60000,784) ).T
        # load test data
        with open(os.path.join(path, 't10k-images.idx3-ubyte')) as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=16)
            self.test_data = np.fromfile(file=fd, dtype=np.uint8).reshape( (10000,784) ).T

        # load training labels
        with open(os.path.join(path,'train-labels.idx1-ubyte')) as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=8)
            data_labels = np.fromfile(file=fd, dtype=np.uint8).reshape( (60000,1) )
        self.teacher = self.prepare_teacher(data_labels,10,cfg)
        # load test labels
        with open(os.path.join(path,'t10k-labels.idx1-ubyte')) as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=8)
            test_labels = np.fromfile(file=fd, dtype=np.uint8).reshape( (10000,1) )
        self.test_teacher = self.prepare_teacher(test_labels,10,cfg)

class MNISTOneMinusData(DataSet):
    def __init__(self,cfg, path):
        cfg.px=cfg.py=28
        cfg.num_classes =10
        cfg.test_batchsize=16*25
        cfg.maps_bottom=1

        if cfg.batchsize==-1:
            cfg.batchsize=16*25
        # load training data
        with open(os.path.join(path,'train-images.idx3-ubyte')) as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=16)
            self.data = (255-np.fromfile(file=fd, dtype=np.uint8).reshape( (60000,784) ).T).astype(np.uint8)
        # load test data
        with open(os.path.join(path, 't10k-images.idx3-ubyte')) as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=16)
            self.test_data = (255 - np.fromfile(file=fd, dtype=np.uint8).reshape( (10000,784) ).T).astype(np.uint8)
        # load training labels
        with open(os.path.join(path,'train-labels.idx1-ubyte')) as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=8)
            data_labels = np.fromfile(file=fd, dtype=np.uint8).reshape( (60000,1) )
        self.teacher = self.prepare_teacher(data_labels,10,cfg)
        # load test labels
        with open(os.path.join(path,'t10k-labels.idx1-ubyte')) as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=8)
            test_labels = np.fromfile(file=fd, dtype=np.uint8).reshape( (10000,1) )
        self.test_teacher = self.prepare_teacher(test_labels,10,cfg)

class MNISTTwiceData(DataSet):
    def __init__(self,cfg,path):
        cfg.maps_bottom=2
        cfg.px=cfg.py=28
        if cfg.batchsize==-1:
            cfg.batchsize=16*25
        with open(os.path.join(path,'train-images.idx3-ubyte')) as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=16)
            data = np.fromfile(file=fd, dtype=np.uint8).reshape( (60000,784)).T
            self.data = np.vstack((data,data))
        # load training labels
        with open(os.path.join(path,'train-labels.idx1-ubyte')) as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=8)
            data_labels = np.fromfile(file=fd, dtype=np.uint8).reshape( (60000,1) )
        self.teacher = self.prepare_teacher(data_labels,10,cfg)
        # load test labels
        with open(os.path.join(path,'t10k-labels.idx1-ubyte')) as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=8)
            test_labels = np.fromfile(file=fd, dtype=np.uint8).reshape( (10000,1) )
        self.test_teacher = self.prepare_teacher(test_labels,10,cfg)

class MNISTTestData(DataSet):
    def __init__(self,cfg,path):
        cfg.px=cfg.py=28
        if cfg.batchsize==-1:
            cfg.batchsize=16*25
        with open(os.getenv("HOME")+'/MNIST/t10k-images.idx3-ubyte') as fd:
            np.fromfile(file=fd, dtype=np.uint8, count=16)
            data = np.fromfile(file=fd, dtype=np.uint8).reshape( (10000,784) )

class ImagePatchesData(DataSet):
    def __init__(self,cfg,path):
        cfg.px=cfg.py=28
        if cfg.batchsize==-1:
            cfg.batchsize=128
        self.data = np.load(os.path.join(path, "prog/patches/patches-kyo-60000-28-28.npy"))
        self.data += 0.5
        self.data *= 128 # now between 0 and 255
        self.data = self.data.astype("uint8")
        self.teacher = self.data
        #self.logtransform()

class CaltechData(DataSet):
    def __init__(self,cfg, path, color, test_batch_num, prefix=""):
        if cfg.batchsize==-1:
            cfg.batchsize=128
        if len(prefix):
            cfg.num_classes    = 1
            cfg.test_batchsize = 9
            cfg.batchsize      = 27
        else:
            cfg.test_batchsize = 102
            cfg.num_classes    = 102
        cfg.px = cfg.py    = 128
        cfg.utype[0]       = pyrbm.UnitType.gaussian
        cfg.maps_bottom    = [1,3][color=="color"]
        D,L = [],[]
        for i in xrange(10): # TODO: use all 10
            if i == test_batch_num: continue
            fn = os.path.join(path ,       "%strain-%s-batch%d.npy"%(prefix,color,i))
            D.append(np.load(fn))
            fn = os.path.join(path , "%slabel-train-%s-batch%d.npy"%(prefix,color,i))
            L.append(np.load(fn))
        self.data    = np.vstack(D).T
        data_labels  = np.hstack(L)
        self.test_data    = np.load(os.path.join(path,       "%svalidation-%s-batch%d.npy" % (prefix,color,test_batch_num))).T


        #self.test_data    = np.load(os.path.join(path,       "test-%s-batch%d.npy" % (color,test_batch_num))).T
        test_labels       = np.load(os.path.join(path, "%slabel-validation-%s-batch%d.npy" % (prefix,color,test_batch_num))).T
        #test_labels       = np.load(os.path.join(path, "label-test-%s-batch%d.npy" % (color,test_batch_num))).T

        if len(prefix):
            data_labels[:] = 0
            test_labels[:] = 0

        self.teacher = self.prepare_teacher(data_labels,cfg.num_classes,cfg)
        self.test_teacher = self.prepare_teacher(test_labels, cfg.num_classes,cfg)
    def dump(self,path,data,labels):
        idx = 0
        for i in xrange(data.shape[1]):
            p = os.path.join(path, str(labels[i]))
            if not os.path.exists(p):
                os.makedirs(p)
            pix = data[:,i].reshape(128,128)
            img = Image.fromarray(np.uint8(pix))
            img.save(os.path.join(p,str(idx) + ".jpg"))
            idx+=1

