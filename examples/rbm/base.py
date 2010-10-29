import sets
import re
import time
import cPickle
import sys, os
import pyublas
import threading
import cuv_python as cp
import numpy as np
import math
from minibatch_provider import MiniBatchProviderEmpty
#import matplotlib.pyplot as plt

if False:
    def __cpy(x):
        y = cp.dev_matrix_cmf(x.h,x.w)
        cp.apply_binary_functor(y, x, cp.binary_functor.COPY)
        return y
    def __pushhost(x):
        y = cp.dev_matrix_cmf_orig(x.h,x.w)
        cp.convert(y,x)
        return y

def first_pool_zeros(images,factor,maps):
    images_rm   = cp.make_rm_view(images,images.w*maps,images.h/maps)
    cp.first_pool_zeros(images_rm,factor)

# debugging libs

from helper_functions import *
from helper_classes import *
import minibatch_provider

gTemp = 1

def max_pool(L,sample_bottom=False,sample_top=False):
    if L.act == L.act_supersampled:
        if sample_bottom or sample_top: L.sample()
        return
    big         = L.act_supersampled
    small       = L.act
    maps        = L.maps
    factor      = L.factor
    big_rm      = cp.make_rm_view(big,big.w*maps,big.h/maps)
    small_rm    = cp.make_rm_view(small,small.w*maps,small.h/maps)
    maxpooltype = L.maxpooltype
    if maxpooltype in [MaxPoolType.first , MaxPoolType.plain]:
        cp.max_pool(small_rm,big_rm,factor,overlap=0)
        if sample_bottom: L.sample_bottom()
        if sample_top:    L.sample()
    elif maxpooltype == MaxPoolType.soft:
        cp.prob_max_pooling(small_rm.vec,big_rm,factor,sample=(sample_bottom or sample_top))

def supersample(L):
    if L.act == L.act_supersampled:
        return
    big         = L.act_supersampled
    small       = L.act
    maps        = L.maps
    factor      = L.factor
    big_rm      = cp.make_rm_view(big,big.w*maps,big.h/maps)
    small_rm    = cp.make_rm_view(small,small.w*maps,small.h/maps)
    maxpooltype = L.maxpooltype

    if maxpooltype == MaxPoolType.first:
        numberBinsPerImage = big.h / maps / factor**2
        indices = cp.dev_matrix_rmi(big.w * maps, numberBinsPerImage )
        cp.fill(indices.vec,0)
        cp.super_to_max(big_rm,small_rm,factor,0,indices)
        indices.dealloc()
    else:
        #cp.apply_scalar_functor(small_rm.vec,cp.scalar_functor.MULT,1./factor**2)
        cp.supersample(big_rm,small_rm,factor)
        #cp.apply_scalar_functor(small_rm.vec,cp.scalar_functor.MULT,   factor**2)

def isnan(x):
    if np.isnan(cp.pull(x).sum()): 
        print("asdf")
        pdb.set_trace()

class WeightLayer(object):
    def __init__(self,layer1,layer2,cfg,layernum):
        self.mat=cp.dev_matrix_cmf(layer1.size,layer2.size)
        #W_       = (np.random.standard_normal((size1,size2))/10).astype('float32').copy('F')
        #W_       = (np.eye(layer1.size)).astype('float32').copy('F')
        #self.mat=cp.push(W_)
        cp.fill(self.mat,0)
        cp.add_rnd_normal(self.mat)
        fact = 1.0
        if layer2.unit_type == UnitType.binary or layer1.unit_type == UnitType.binary:
            # the 0.5 stems from the fact that our upper layer has activation 0.5 on average, not 0, if we use binary hidden units.
            fact = 0.5

        cp.apply_scalar_functor(self.mat,cp.scalar_functor.MULT,
                                fact/math.sqrt(max(layer1.size, layer2.size)))
        self.allocBias(layer1,layer2)
        self.num_params = self.mat.h*self.mat.w + len(self.bias_lo) + len(self.bias_hi)

    def allocBias(self,layer1,layer2):
        self.bias_lo=cp.dev_matrix_cmf(layer1.size,1)
        self.bias_hi=cp.dev_matrix_cmf(layer2.size,1)
        cp.fill(self.bias_lo,0)
        cp.fill(self.bias_hi,0)
    def save(self,prefix,postfix):
        np.save(os.path.join(prefix,"weights-%s.npy"%postfix),cp.pull(self.mat))
        np.save(os.path.join(prefix,"bias-lo-%s.npy"%postfix),cp.pull(self.bias_lo))
        np.save(os.path.join(prefix,"bias-hi-%s.npy"%postfix),cp.pull(self.bias_hi))
    def load(self,prefix,postfix):
        fn = os.path.join(prefix,"weights-%s.npy"%postfix)
        if os.path.exists(fn):
            self.mat.dealloc()
            self.mat=cp.push(np.load(fn))
            self.bias_lo.dealloc()
            self.bias_hi.dealloc()
            self.bias_lo = cp.push(np.load(os.path.join(prefix,"bias-lo-%s.npy"%postfix)))
            self.bias_hi = cp.push(np.load(os.path.join(prefix,"bias-hi-%s.npy"%postfix)))
    def downPass(self,layer1,layer2,sample,supersample):
        cp.prod(layer1.act,self.mat,layer2.act,'n','n')
        layer1.postUpdateFromAbove(sample,bias=self.bias_lo,do_supersample=supersample)
    def upPass(self,layer1,layer2,sample,blur):
        cp.prod(layer2.act,self.mat,layer1.act,'t','n')
        layer2.postUpdateFromBelow(sample,bias=self.bias_hi)
    def allocUpdateMatrix(self):
        self.w_tmp =cp.dev_matrix_cmf(self.mat.h,self.mat.w)
        cp.fill(self.w_tmp.vec,0)
        self.blo_tmp=cp.dev_matrix_cmf(len(self.bias_lo),1)
        self.bhi_tmp=cp.dev_matrix_cmf(len(self.bias_hi),1)
        cp.fill(self.blo_tmp,0)
        cp.fill(self.bhi_tmp,0)
    def deallocUpdateMatrix(self):
        cp.safeThreadSync()
        if "w_tmp" in self.__dict__:
            self.w_tmp.dealloc()
        if "blo_tmp" in self.__dict__:
            self.blo_tmp.dealloc()
        if "bhi_tmp" in self.__dict__:
            self.bhi_tmp.dealloc()
    def updateStep(self,learnrate,cost):
        cp.learn_step_weight_decay(self.mat.vec,self.w_tmp.vec,learnrate,cost) # W  += learnrate(dW - cost*W)
        cp.learn_step_weight_decay(self.bias_lo,self.blo_tmp,learnrate,cost) # W  += learnrate(dW - cost*W)
        cp.learn_step_weight_decay(self.bias_hi,self.bhi_tmp,learnrate,cost) # W  += learnrate(dW - cost*W)
        #cp.apply_scalar_functor(self.bias_hi,cp.scalar_functor.ADD,-0.00001)
        #cp.apply_scalar_functor(self.bias_lo,cp.scalar_functor.ADD,-0.00001)
    def updateGradientNeg(self,layer1,layer2,batchsize):
        cp.prod(self.w_tmp,layer1.act,layer2.act,'n','t',-1./batchsize,1./batchsize)
        cp.reduce_to_col(self.blo_tmp.vec,layer1.act,cp.reduce_functor.ADD,-1./batchsize,1./batchsize)
        cp.reduce_to_col(self.bhi_tmp.vec,layer2.act,cp.reduce_functor.ADD,-1./batchsize,1./batchsize)
    def updateGradientPos(self,layer1,layer2):
        cp.prod(self.w_tmp,layer1.act,layer2.act,'n','t')
        cp.reduce_to_col(self.blo_tmp.vec,layer1.act)
        cp.reduce_to_col(self.bhi_tmp.vec,layer2.act)


class DiaWeightLayer(WeightLayer):
    def __init__(self,layer1,layer2,cfg,layernum,sauce=True):
        fs = cfg.local_patchsize[layernum] # filter size
        nm = layer2.maps      # number of maps
        rf = cfg.local_steepness # steepness of diagonals
        print "Creating ", nm, " maps...",
        px = int(np.sqrt(layer1.size/layer1.maps))
        self.mat_ff = cp.filter_factory_float(px,px,fs,layer1.maps,layer2.maps)
        #msize = fs*fs*(nm+layer1.maps-1)/rf
        #off = []
        #px = int(np.sqrt(layer1.size/layer1.maps))
        #for m in xrange(nm+layer1.maps-1):
        #    for i in xrange(fs):
        #        for j in xrange(fs/rf):
        #            #assert i*px/rf+j < layer2.size/layer2.maps
        #            off.append(i*px/rf+j + m*layer2.size_supersampled/layer2.maps)
        #assert msize == len(off)
        #for i in xrange(msize):
        #    off[i] += - (px+1)*(int(fs/2))/rf - (layer1.maps-1)*layer1.size/layer1.maps/rf
        ##off2 = sets.Set(off)
        #off = sorted(sets.Set(off))
        ##assert len(off2) == len(off)
        #self.mat = cp.dev_dia_matrix_f(layer1.size,layer2.size_supersampled,off,layer1.size,rf) # watch out, only horizontally longer matrices are supported!
        mat_host = self.mat_ff.get_dia()
        self.mat = cp.dev_dia_matrix_f()
        cp.convert(self.mat,mat_host)
        mat_host.dealloc()
        self.bd  = cp.dev_block_descriptor_f(self.mat)
        cp.fill(self.mat.vec,0)
        cp.add_rnd_normal(self.mat.vec)
        num_inputs = layer2.maps * fs*fs
        fact = 1.0
        if layer2.unit_type == UnitType.binary or layer1.unit_type == UnitType.binary:
            # the 0.5 stems from the fact that our upper layer has activation 0.5 on average, not 0, if we use binary hidden units.
            fact = 0.5
        cp.apply_scalar_functor(self.mat.vec,cp.scalar_functor.MULT,fact/math.sqrt(num_inputs))
        #noise=np.random.normal(0,1,self.mat.vec.size())
        #noise_dev=cp.push(noise.reshape(len(noise),1).copy('F').astype('float32'))
        #cp.apply_binary_functor(self.mat.vec,noise_dev.vec,cp.binary_functor.COPY)
        self.allocBias(layer1,layer2)
        self.num_params = self.mat.stride * self.mat.num_dia + len(self.bias_lo) + len(self.bias_hi)
        print "done."
        if not cfg.maxpooltype==MaxPoolType.first and sauce:
            self.allocSauceMat(layer2,cfg,fs)
    def allocSauceMat(self, layer2, cfg,weights_fs):
        print("creating gaussian filter for hidden activations")
        std = 0.05 * math.sqrt(layer2.size_supersampled/layer2.maps / float(weights_fs))
        #std = 0.2
        fs = int(2*math.ceil(std)+1)
        print "Sauce matrix: std=%2.3f, fs=%2.3f" % (std,fs)
        nm = layer2.maps      # number of maps
        msize = fs*fs
        off = []
        px=int(np.sqrt(layer2.size_supersampled/layer2.maps))
        #for m in xrange(nm+layer2.maps-1):
        for i in xrange(fs):
            for j in xrange(fs):
                #off.append(i*px+j + m*layer2.size/layer2.maps)
               off.append(i*px+j)
        assert msize == len(off)
        for i in xrange(msize):
            off[i] += - (px+1)*(fs-1)/2 # - (layer2.maps-1)*layer2.size/layer2.maps
        self.latMat = cp.dev_dia_matrix_f(layer2.size_supersampled,layer2.size_supersampled,off,layer2.size_supersampled,1) # watch out, only horizontally longer matrices are supported!
        cp.fill(self.latMat.vec,0)
        stds = {}
        for o2 in off:
            o = abs(o2)
            val = (o / px)**2 + min(o % px, px - (o%px))**2
            val2 = 1./(math.sqrt(2.*3.14)*std) * math.exp(-0.5 * val/std**2)
            stds[o2] = val2
        norm = sum(stds.values())

        for o2 in off:
            #print " ---------------------> (o=%d, x=%d, y=%d, val=%3.4f)"%( o2, (o/px),min(o % px, px - (o%px)), val2)
            vec = self.latMat.dia(o2)
            cp.fill( vec, stds[o2] )
            # now make sure that we do not blur across maps
            for i in xrange(vec.size):
               h = i % px**2
               if   o2 < 0 and h<-o2:   vec.set(i,0)
               elif o2 > 0 and h>px**2-o2: vec.set(i,0)
        m = self.latMat
        dias = map(lambda x:m.dia(x), off)
        for i in xrange(m.h):
           s = 0
           for dia in dias: s += dia(i)
           for dia in dias: dia.set(i, dia(i)/s)

    def allocBias(self,layer1,layer2):
        self.bias_lo=cp.dev_matrix_cmf(layer1.size,1)
        self.bias_hi=cp.dev_matrix_cmf(layer2.size_supersampled,1)
        cp.fill(self.bias_lo,0)
        cp.fill(self.bias_hi,0)
    def downPass(self,layer1,layer2,sample,supersample):
        cp.prod(layer1.act,self.mat,layer2.act_supersampled,'n','n')
        layer1.postUpdateFromAbove(sample,bias=self.bias_lo,do_supersample=supersample)
    def upPass(self,layer1,layer2,sample,blur):
        cp.prod(layer2.act_supersampled,self.mat,layer1.act,'t','n')
        layer2.postUpdateFromBelow(sample, bias=self.bias_hi)
        if blur:
            tmp = get_copy(layer2.act_supersampled)
            cp.prod(layer2.act_supersampled, self.latMat, tmp, 'n','n')
            tmp.dealloc()
    def allocUpdateMatrix(self):
        self.w_tmp=cp.dev_dia_matrix_f(self.mat)
        cp.fill(self.w_tmp.vec,0)
        self.blo_tmp=cp.dev_matrix_cmf(len(self.bias_lo),1)
        self.bhi_tmp=cp.dev_matrix_cmf(len(self.bias_hi),1)
        cp.fill(self.blo_tmp,0)
        cp.fill(self.bhi_tmp,0)
    def deallocUpdateMatrix(self):
        self.w_tmp.dealloc()
        self.blo_tmp.dealloc()
        self.bhi_tmp.dealloc()
    def updateGradientNeg(self,layer1,layer2,batchsize):
        cp.densedense_to_dia(self.w_tmp, self.bd, layer1.act, layer2.act_supersampled,-1./batchsize, 1./batchsize)
        cp.reduce_to_col(self.blo_tmp.vec,layer1.act,cp.reduce_functor.ADD,-1./batchsize,1./batchsize)
        cp.reduce_to_col(self.bhi_tmp.vec,layer2.act_supersampled,cp.reduce_functor.ADD,-1./batchsize,1./batchsize)
    def updateGradientPos(self,layer1,layer2):
        cp.densedense_to_dia(self.w_tmp, self.bd, layer1.act, layer2.act_supersampled)
        cp.reduce_to_col(self.blo_tmp.vec,layer1.act)
        cp.reduce_to_col(self.bhi_tmp.vec,layer2.act_supersampled)
    def save(self,prefix,postfix):
        self.mat.save(os.path.join(prefix,"weights-%s.npy"%postfix))
        np.save(os.path.join(prefix,"bias-lo-%s.npy"%postfix),cp.pull(self.bias_lo))
        np.save(os.path.join(prefix,"bias-hi-%s.npy"%postfix),cp.pull(self.bias_hi))
    def load(self,prefix,postfix):
        fn = os.path.join(prefix,"weights-%s.npy"%postfix)
        if os.path.exists(fn):
            self.mat.dealloc()
            self.mat.load(fn)
            self.bd  = cp.dev_block_descriptor_f(self.mat)
            self.bias_lo.dealloc()
            self.bias_hi.dealloc()
            self.bias_lo = cp.push(np.load(os.path.join(prefix,"bias-lo-%s.npy"%postfix)))
            self.bias_hi = cp.push(np.load(os.path.join(prefix,"bias-hi-%s.npy"%postfix)))

class DiaLateralWeightLayer(DiaWeightLayer):
    def __init__(self,layer1,layer2,cfg,layernum):
        DiaWeightLayer.__init__(self,layer1,layer2,cfg,layernum,sauce=False)
        fs = cfg.lateral_patchsize # filter size
        nm = layer2.maps      # number of maps
        size_supersampled=layer1.size*nm
        msize = fs*fs
        off = []
        px=int(np.sqrt(layer1.size/layer1.maps))
        #for m in xrange(nm+layer2.maps-1):
        for i in xrange(fs):
            for j in xrange(fs):
                #off.append(i*px+j + m*layer2.size/layer2.maps)
                off.append(i*px+j)
        assert msize == len(off)
        for i in xrange(msize):
            off[i] += - (px+1)*(fs-1)/2 # - (layer2.maps-1)*layer2.size/layer2.maps
        self.latMat = cp.dev_dia_matrix_f(size_supersampled,size_supersampled,off,size_supersampled,1) # watch out, only horizontally longer matrices are supported!
        self.latBd  = cp.dev_block_descriptor_f(self.latMat)
        cp.fill(self.latMat.vec,0)
        off = sorted(sets.Set(off))
    def save(self,prefix,postfix):
        DiaWeightLayer.save(self,prefix,postfix)
        self.latMat.save(os.path.join(prefix,"lat_weights-%s.npy"%postfix))
    def load(self,prefix,postfix):
        DiaWeightLayer.load(self,prefix,postfix)
        fn = os.path.join(prefix,"lat_weights-%s.npy"%postfix)
        if os.path.exists(fn):
            self.latMat.dealloc()
            self.latMat.load(fn)
            self.latBd  = cp.dev_block_descriptor_f(self.latMat)
    def upPass(self,layer1,layer2,sample,blur):
        #blur is only maintained for compatibility, never used

        #DiaWeightLayer.upPass(self,layer1,layer2)

        ### calculate bottum up input
        cp.prod(layer2.act_supersampled,self.mat,layer1.act,'t','n')

        # bottom-up input w/o bias(!)
        bui = cp.dev_matrix_cmf(layer2.act_supersampled.h,layer2.act_supersampled.w)
        copy(bui,layer2.act_supersampled)

        # the ``last'' state for lateral iteration, just set to sigmoid(bui) for initialization
        layer2.postUpdateFromBelow(sample,self.bias_hi)                               # sigmoid
        laststate = cp.dev_matrix_cmf(layer2.act_supersampled.h,layer2.act_supersampled.w)   
        copy(laststate,layer2.act_supersampled)
        #self.pull_requested("before lat")

        for step in xrange(5):    
            # x = I * x_old
            cp.prod(layer2.act_supersampled,self.latMat,laststate,'t','n')
            # x = sigm(x + Wy)   (add bottom up input)
            cp.apply_binary_functor(layer2.act_supersampled,bui,cp.binary_functor.ADD)
            layer2.postUpdateFromBelow(sample,self.bias_hi)                               # sigmoid

            # x = 0.8 * x_old  + 0.2 * x
            cp.apply_binary_functor(layer2.act_supersampled,laststate,cp.binary_functor.AXPBY,0.8,0.2)
            copy(laststate,layer2.act_supersampled)
        laststate.dealloc()
        bui.dealloc()
        #self.pull_requested("after lat")

    def allocUpdateMatrix(self):
        DiaWeightLayer.allocUpdateMatrix(self)
        self.lat_w_tmp=cp.dev_dia_matrix_f(self.latMat)
        cp.fill(self.lat_w_tmp.vec,0)
    def deallocUpdateMatrix(self):
        DiaWeightLayer.deallocUpdateMatrix(self)
        self.lat_w_tmp.dealloc()
    def updateStep(self,learnrate,cost):
        DiaWeightLayer.updateStep(self,learnrate,cost)
        cp.learn_step_weight_decay(self.latMat.vec,self.lat_w_tmp.vec,learnrate,cost) 
        cp.fill(self.latMat.dia(0), 0)
    def updateGradientNeg(self,layer1,layer2,batchsize):
        DiaWeightLayer.updateGradientNeg(self,layer1,layer2,batchsize)
        cp.densedense_to_dia(self.lat_w_tmp, self.latBd, layer2.act_supersampled, layer2.act_supersampled,-1./batchsize, 1./batchsize)
    def updateGradientPos(self,layer1,layer2):
        DiaWeightLayer.updateGradientPos(self,layer1,layer2)
        cp.densedense_to_dia(self.lat_w_tmp, self.latBd, layer2.act_supersampled, layer2.act_supersampled)

class NodeLayer(object):
    def __init__(self, size, batchsize, unit_type,sigma,maps,factor,maxpooltype):
        self.size = size/factor**2
        self.size_supersampled = size
        self.bsize = batchsize
        self.unit_type = unit_type
        self.maps = maps
        self.factor=factor
        self.sigma=sigma
        self.alloc()
        self.maxpooltype=maxpooltype
    def sample(self):
        if self.unit_type == UnitType.gaussian or self.unit_type == UnitType.cont:
            cp.add_rnd_normal(self.act)
        elif self.unit_type == UnitType.binary:
            cp.rnd_binarize(self.act)
    def sample_bottom(self):
        if self.unit_type == UnitType.gaussian or self.unit_type == UnitType.cont:
            cp.add_rnd_normal(self.act_supersampled)
        elif self.unit_type == UnitType.binary:
            cp.rnd_binarize(self.act_supersampled)
    def alloc(self):
        self.act = cp.dev_matrix_cmf(self.size,self.bsize)
        cp.fill(self.act,0)
        if self.factor != 1:
            self.act_supersampled = cp.dev_matrix_cmf(self.size_supersampled,self.bsize)
            cp.fill(self.act_supersampled,0)
        else:
            self.act_supersampled = self.act
        return self
    def dealloc(self):
        self.act.dealloc()
        if self.factor != 1:
            self.act_supersampled.dealloc()
    def nonlinearity(self):
        self.__nonlinearity(self.act_supersampled)
    def nonlinearity_top(self):
        self.__nonlinearity(self.act)
    def __nonlinearity(self,dst):
        if not self.unit_type == UnitType.gaussian:
            #cp.apply_scalar_functor(dst,cp.scalar_functor.MULT,-1)
            cp.apply_scalar_functor(dst,cp.scalar_functor.SIGM)
    def allocPChain(self):
        self.pchain=cp.dev_matrix_cmf(self.size, self.bsize)
        cp.fill(self.pchain,0)
        if self.factor!=1:
            self.pchain_supersampled=cp.dev_matrix_cmf(self.size_supersampled, self.bsize)
            cp.fill(self.pchain_supersampled,0)
        else:
            self.pchain_supersampled=self.pchain
    def deallocPChain(self):
        if not "pchain" in self.__dict__:
            return
        self.pchain.dealloc()
        if self.factor!=1:
            self.pchain_supersampled.dealloc()
    def switchToPChain(self):
        self.org, self.org_supersampled = self.act, self.act_supersampled
        self.act, self.act_supersampled = self.pchain, self.pchain_supersampled
    def switchToOrg(self):
        self.pchain, self.pchain_supersampled = self.act, self.act_supersampled
        self.act, self.act_supersampled       = self.org, self.org_supersampled
    def postUpdateFromAbove(self,sample,bias,do_supersample=False):
        cp.matrix_plus_col(self.act,bias.vec)
        self.nonlinearity_top()
        if sample:
            self.sample()
        if do_supersample:
            supersample(self)
    def postUpdateFromBelow(self,sample,bias):
        cp.matrix_plus_col(self.act_supersampled,bias.vec)
        self.nonlinearity()
        if self.maxpooltype==MaxPoolType.first and self.factor!=1:
            first_pool_zeros(self.act_supersampled,self.factor,self.maps)
        if sample:
            self.sample_bottom()


class RBMStack:
    """ Represents a stack of Restricted Boltzmann Machines """
    def __init__(self, cfg):
        """ Creates a stack of RBMs with cfg.num_layers levels. Each Layer is initialized using cfg."""
        self.layers = []
        self.weights= []
        self.cfg = cfg;
        self.states = {}

        for layer in xrange(self.cfg.num_layers):
            if layer==0:
                local_pooling=1
            else:
                local_pooling=cfg.local_pooling
            maps = cfg.local_maps[layer]
            assert(not self.cfg.l_size[layer] % maps)
            self.layers.append(NodeLayer(self.cfg.l_size[layer], self.cfg.batchsize,self.cfg.utype[layer],cfg.sigma,maps,local_pooling,self.cfg.maxpooltype))
            if self.cfg.cd_type == CDType.pcd:
                self.layers[layer].allocPChain()

        if self.cfg.srbmhid and self.cfg.local:
            weight_type=DiaLateralWeightLayer
        elif self.cfg.srbmhid:
            weight_type=LateralWeightLayer
        elif self.cfg.local:
            weight_type=DiaWeightLayer
        else:
            weight_type=WeightLayer

        for layer in xrange(self.cfg.num_layers-1):
            self.weights.append(weight_type(self.layers[layer],self.layers[layer+1],self.cfg, layernum=layer))

        self.reconstruction_error = []
        self.matrequests_new = {}
        self.matrequests     = {}
        self.pulled = {}
    def request_mats(self, L, eventobj):
        self.matrequests.clear()
        self.matrequests_new.clear()
        for m in L:
            self.matrequests_new[m] = eventobj
    def request_mat(self, name, eventobj):
        self.matrequests_new[name] = eventobj
    def name2prop(self,name):
        p = {}
        p["name"] = name
        try:
            L = name.split("->")
            p["layer"] = int(re.search('Layer(\d+)',L[0]).group(1))
            p["state"] = str(L[1])
            p["what"]  = str(L[2])
            p["index"] = int(str(L[3]))
        except:
            pass
        return p
    def name2mat(self, name, state):
        #print "Request for ", name
        if name == "reconstruction_error":
            return np.array(self.reconstruction_error)
        elif name.startswith("Dataset->mean"):
            return -cp.pull(self.mbp.mbs.negative_mean).reshape(self.cfg.px,self.cfg.py)
        elif name.startswith("Dataset->range"):
            return cp.pull(self.mbp.mbs.range).reshape(self.cfg.px,self.cfg.py)
        elif name.startswith("Dataset->min"):
            return -cp.pull(self.mbp.mbs.negative_min).reshape(self.cfg.px,self.cfg.py)
        elif name.startswith("Dataset->max"):
            return (-cp.pull(self.mbp.mbs.negative_min) + cp.pull(self.mbp.mbs.range)).reshape(self.cfg.px,self.cfg.py)
        elif name.startswith("Dataset->std"):
            return cp.pull(self.mbp.mbs.std).reshape(self.cfg.px,self.cfg.py)
        p = self.name2prop(name)
        if p["state"] != state:
            return None
        if p["index"] >= self.cfg.batchsize and p["what"] != "Weights":
            return None
        if False: pass
        elif p["what"] == "BiasHi":
            W = self.weights[p["layer"]]
            lo = self.layers[p["layer"]+0]
            hi = self.layers[p["layer"]+1]
            mat = cp.pull(W.bias_hi)
            px = int(math.sqrt(hi.size_supersampled/hi.maps))
            if mat.shape[0] == px*px*hi.maps:
                mat = mat[:,p["index"]].reshape(px*hi.maps,px)
        elif p["what"] == "BiasLo":
            W = self.weights[p["layer"]]
            lo = self.layers[p["layer"]+0]
            hi = self.layers[p["layer"]+1]
            mat = cp.pull(W.bias_lo)
            px = int(math.sqrt(lo.size/lo.maps))
            if mat.shape[0] == px*px*lo.maps:
                mat = mat[:,p["index"]].reshape(px*lo.maps,px)
        elif p["what"] == "PChain":
            L = self.layers[p["layer"]]
            mat = cp.pull(L.pchain_supersampled)
            px = int(math.sqrt(L.size_supersampled/L.maps))
            if mat.shape[0] == px*px*L.maps:
                mat = mat[:,p["index"]].reshape(px*L.maps,px)
        elif p["what"] == "Sub PChain":
            L = self.layers[p["layer"]]
            mat = cp.pull(L.pchain)
            px = int(math.sqrt(L.size/L.maps))
            if mat.shape[0] == px*px*L.maps:
                mat = mat[:,p["index"]].reshape(px*L.maps,px)
        elif p["what"] == "Act":
            L = self.layers[p["layer"]]
            mat = cp.pull(L.act_supersampled)
            px = int(math.sqrt(L.size_supersampled/L.maps))
            if mat.shape[0] == px*px*L.maps:
                mat = mat[:,p["index"]].reshape(px*L.maps,px)
        elif p["what"] == "AllWeights":
            W  = self.weights[p["layer"]]
            if W.__class__ == WeightLayer:
                return None
            lo = self.layers[p["layer"]]
            hi = self.layers[p["layer"]+1]
            I  = p["index"]
            px = int(math.sqrt(hi.size_supersampled/hi.maps))
            baseidx = I*px*px
            fs = self.cfg.local_patchsize[p["layer"]] # filter size
            mat = np.zeros((fs*px*lo.maps,fs*px))
            if hi.size_supersampled == px*px*hi.maps:
                for i in xrange(px):
                    for j in xrange(px):
                        filt = W.mat_ff.extract_filter(W.mat, baseidx + i*px+j)
                        filt = cp.pull(filt).reshape(fs*lo.maps,fs)
                        for k in xrange(lo.maps):
                            mb = k*fs*px
                            fb = k*fs
                            mat[ mb + fs*i : mb + fs*i+fs,    fs*j : fs*j+fs] = filt[fb:fs+fb,:]
        elif p["what"] == "Weights":
            W = self.weights[p["layer"]]
            L = self.layers[p["layer"]]
            I = p["index"]
            if W.__class__ == WeightLayer:
                mat = cp.pull(W.mat)
                mat = mat[:,I]
                px = int(math.sqrt(L.size/L.maps))
                mat = mat.reshape(px*L.maps,px)
            else:
                li = []
                px = int(math.sqrt(L.size/L.maps))
                for i in xrange(W.mat.h):
                    li.append(W.mat(i,I))
                mat = np.array(li).reshape(px*L.maps,px)

                #mat = W.mat_ff.extract_filter(W.mat, I)
                #fs = self.cfg.local_patchsize[p["layer"]] # filter size
                #mat = cp.pull(mat).reshape(fs*L.maps,fs)
        elif p["what"] == "Sub Act":
            L = self.layers[p["layer"]]
            mat = cp.pull(L.act)
            px = int(math.sqrt(L.size/L.maps))
            if mat.shape[0] == px*px*L.maps:
                mat = mat[:,p["index"]].reshape(px*L.maps,px)
        else:
            return None
        return mat
    def pull_requested(self, state, timeout=0.05):
        self.states[state] = 1
        self.cfg.states = self.states.keys()
        #self.pyro_daemon.handleRequests(0.0001)
        done = {}
        for name in self.matrequests.keys():
            ret = self.name2mat(name,state)
            if ret != None:
                self.pulled[name] = ret
                done[name] = 1
        for name in done.keys():
            if name in self.matrequests.keys():
                self.matrequests[name].set()
        if timeout>0 and len(done.keys()):
            e = threading.Event()
            e.wait(timeout)
        for name in done.keys():
            if name in self.matrequests.keys():
                del self.matrequests[name]

    def getErr(self,layernum,orig_data):
        cp.apply_binary_functor(self.layers[layernum].act,orig_data,cp.binary_functor.SUBTRACT)
        sqerr = cp.norm2(self.layers[layernum].act)**2
        return sqerr/((self.layers[layernum].size)*self.cfg.batchsize)
    def saveLayer(self,layernum,prefix,postfix):
        self.weights[layernum].save(prefix,str(layernum)+postfix)
    def loadLayer(self,layernum,prefix,postfix):
        self.weights[layernum].load(prefix,str(layernum)+'-'+postfix)
    def upPass(self, layernum,sample=True,blur=False):
        if self.cfg.maxpooltype==MaxPoolType.first:
            blur=False
        self.weights[layernum].upPass(self.layers[layernum],self.layers[layernum+1],sample,blur=blur)
    def downPass(self, layernum,sample=True,supersample=False):
        self.weights[layernum-1].downPass(self.layers[layernum-1],self.layers[layernum],sample=sample,supersample=supersample)
    def updateLayer(self,layernum,sample=True,blur=True):
        L = self.layers[layernum]
        if layernum==0:
            self.downPass(layernum+1,sample=sample,supersample=False)
        if layernum==len(self.layers)-1:
            self.upPass(layernum-1,sample)
        if layernum<len(self.layers)-1 and layernum>0: 
            hi = self.layers[layernum+1]
            lo = self.layers[layernum-1]
            wlo = self.weights[layernum-1]
            whi = self.weights[layernum]

            cp.prod(L.act,whi.mat,hi.act_supersampled,'n','n')
            cp.matrix_plus_col(L.act,whi.bias_lo.vec)
            supersample(L)

            tmp = get_copy(L.act_supersampled)
            cp.prod(L.act_supersampled,wlo.mat,lo.act,'t','n')
            cp.matrix_plus_col(L.act_supersampled,wlo.bias_hi.vec)

            # add parts from above/below
            cp.apply_binary_functor(L.act_supersampled,tmp,cp.binary_functor.AXPBY,0.5,0.5)
            tmp.dealloc()

            L.nonlinearity()
            #if blur and L.factor!=1:
            #    tmp2 = get_copy(L.act_supersampled)
            #    cp.prod(L.act_supersampled, wlo.latMat, tmp2, 't','n')
            #    tmp2.dealloc()

            max_pool(L,sample_bottom=sample,sample_top=sample)

    def getLearnrate(self,iter,itermax):
        if self.cfg.learnrate_sched == LearnRateSchedule.linear:
            return self.cfg.learnrate  * (1 - iter/itermax)
        elif self.cfg.learnrate_sched == LearnRateSchedule.exponential:
            return self.cfg.learnrate  * math.exp(-10*iter/itermax)
        elif self.cfg.learnrate_sched == LearnRateSchedule.divide:
            startat = 10000
            if iter < startat:
                return self.cfg.learnrate
            else:
                return self.cfg.learnrate * 1000.0 / (1000.0 +iter - startat)
        else :
            return self.cfg.learnrate

    def resetPChain(self,mbp):
        mbp.forgetOriginalData()
        mbp.getMiniBatch(self.cfg.batchsize,self.layers[0].pchain)
        map(NodeLayer.switchToPChain, self.layers)
        for layernum in xrange(len(self.weights)):
            self.upPass(layernum,True)
        map(NodeLayer.switchToOrg, self.layers)
    def pcdStep(self,layernum,batchsize):
        try:
            self.layers[layernum].switchToPChain()
            self.layers[layernum+1].switchToPChain()
            self.upPass(layernum,sample=True,blur=False)
            self.downPass(layernum+1,sample=True,supersample=False)
            self.weights[layernum].updateGradientNeg(self.layers[layernum],self.layers[layernum+1],self.cfg.batchsize)
        finally:
            self.layers[layernum].switchToOrg()
            self.layers[layernum+1].switchToOrg()

    def cdnStep(self,layernum,batchsize):
        for step in xrange(self.cfg.ksteps):
            self.upPass(layernum,sample=True,blur=False)
            self.downPass(layernum+1,sample=True,supersample=False)
        self.upPass(layernum,sample=True,blur=False)
        self.weights[layernum].updateGradientNeg(self.layers[layernum],self.layers[layernum+1],self.cfg.batchsize)

    def trainDBM(self, mbatch_provider, itermax):
        try:
            """ Train all layers of a RBM-Stack as a DBM using minibatches provided by mbatch_provider for itermax minibatches """
            ### if pcd get starting point for fantasies 
            if self.cfg.cd_type == CDType.pcd:
                self.resetPChain(mbatch_provider)
                mbatch_provider.forgetOriginalData()

            ### temporary matrix to save update
            for weightlayer in self.weights:
                weightlayer.allocUpdateMatrix()

            ### iterate over updates
            for iter in xrange(1,itermax):
                # carry over non-completed requests
                for k in self.matrequests.keys():
                    self.matrequests_new[k] = self.matrequests[k]
                    del self.matrequests[k]
                # add new requests
                self.matrequests, self.matrequests_new = self.matrequests_new, {}
                ### new learnrate if schedule
                learnrate=self.getLearnrate(iter,itermax)/100
                sys.stdout.write('.')
                sys.stdout.flush()
                ### positive phase
                mbatch_provider.getMiniBatch(self.cfg.batchsize,self.layers[0].act)
                self.pull_requested("dbm originals")
                for layernum in xrange(len(self.weights)):
                    self.upPass(layernum,sample=False,blur=False)
                uq = UpdateQ(len(self.layers))
                uq.push([1]) # must start w/ 0-th layer
                while uq.minupdates([0]) < self.cfg.dbm_minupdates:
                    layernum = uq.pop(firstlayer=1)
                    self.updateLayer(layernum,sample=False,blur=True)
                for layernum in xrange(len(self.weights)):
                    self.weights[layernum].updateGradientPos(self.layers[layernum],self.layers[layernum+1])

                ### output stuff
                if iter != 0 and (iter%100) == 0:
                    self.pull_requested("dbm preupdate")
                    self.downPass(1,sample=False,supersample=False)
                    self.pull_requested("dbm postupdate")
                    err=self.getErr(0,mbatch_provider.sampleset)
                    self.reconstruction_error.append(err)
                    print "Iter: ",iter, "Err: %02.06f"%err, "|W|: %02.06f"%cp.norm2(self.weights[0].mat.vec)
                    print self.cfg.workdir,
                    if self.cfg.save_every!=0 and iter % self.cfg.save_every == 0 and iter>0 :
                        for layernum in xrange(len(self.weights)):
                            self.saveLayer(layernum,self.cfg.workdir,"-%010d"%iter)
                #if iter != 0 and (iter%50) == 0:
                #    #print "resetting pchain"
                #    self.resetPChain(mbatch_provider)

                ### negative phase
                ### replace layer nodes with pcd-chain or do initial uppass 
                if self.cfg.cd_type == CDType.cdn:
                    for layernum in xrange(len(self.weights)):
                        self.upPass(layernum,sample=True)
                else:
                    for layer in self.layers:
                        layer.switchToPChain()

                uq = UpdateQ(len(self.layers))
                uq.push([1])
                while uq.minupdates() < self.cfg.dbm_minupdates:
                    layernum = uq.pop(firstlayer=0)
                    self.updateLayer(layernum,sample=True,blur=False)

                ### update gradients
                for layernum in xrange(len(self.weights)):
                    self.weights[layernum].updateGradientNeg(self.layers[layernum],self.layers[layernum+1],self.cfg.batchsize)
                ### update weights and biases
                for weightlayer in self.weights:
                    weightlayer.updateStep(learnrate,self.cfg.cost)

                ### put original layer back in place
                if self.cfg.cd_type == CDType.pcd:
                    for layer in self.layers:
                        layer.switchToOrg()
                mbatch_provider.forgetOriginalData()

        finally:
            for weightlayer in self.weights:
                if "w_tmp" in weightlayer.__dict__:
                    weightlayer.deallocUpdateMatrix()

    def trainLayer(self, mbatch_provider, iterstart, itermax, layernum): # 
        """ Train one layer of a RBM-Stack using minibatches provided by mbatch_provider for itermax minibatches """
        try:
            ### if pcd get starting point for fantasies 
            if self.cfg.cd_type == CDType.pcd:
                mbatch_provider.getMiniBatch(self.cfg.batchsize,self.layers[layernum].pchain)
                mbatch_provider.forgetOriginalData()
            ### temporary matrix to save update
            self.weights[layernum].allocUpdateMatrix()
            ### iterate over updates
            for iter in xrange(iterstart,itermax):
                self.current_iter=iter
                # carry over non-completed requests
                for k in self.matrequests.keys():
                    self.matrequests_new[k] = self.matrequests[k]
                    del self.matrequests[k]
                # add new requests
                self.matrequests, self.matrequests_new = self.matrequests_new, {}
                ### new learnrate if schedule
                learnrate=self.getLearnrate(iter,itermax)
                sys.stdout.write('.')
                sys.stdout.flush()
                ### positive phase
                mbatch_provider.getMiniBatch(self.cfg.batchsize,self.layers[layernum].act)
                self.pull_requested("originals")
                self.upPass(layernum,sample=False,blur=True)
                self.weights[layernum].updateGradientPos(self.layers[layernum],self.layers[layernum+1])
                ### negative phase
                if self.cfg.cd_type == CDType.cdn:
                    self.cdnStep(layernum,self.cfg.batchsize)
                elif self.cfg.cd_type == CDType.pcd:
                    self.pcdStep(layernum,self.cfg.batchsize)
                ### update weights and biases
                self.weights[layernum].updateStep(learnrate,self.cfg.cost)
                self.pull_requested("afterweightupdate")
                if iter != 0 and (iter%100) == 0:
                    self.save(layernum,iter,mbatch_provider)
                    print self.cfg.workdir,
        finally:
            if "w_tmp" in self.weights[layernum].__dict__:
                self.weights[layernum].deallocUpdateMatrix()
                mbatch_provider.forgetOriginalData()

    def save(self,layer, iter,mbatch_provider):

        mbatch_provider.forgetOriginalData()
        mbatch_provider.getMiniBatch(self.cfg.batchsize,self.layers[layer].act)
        supersample(self.layers[layer])

        for l in reversed(xrange(1,layer+1)):
            self.downPass(l,sample=False,supersample=True)
        self.pull_requested("beforesave")
        self.upPass(layer,sample=False);     
        self.downPass(layer+1,sample=False,supersample=True);     
        for l in reversed(xrange(1,layer+1)):
            self.downPass(l,sample=False,supersample=True)
        self.pull_requested("aftersave")

        ### increase temperature, run PCD-chain a bit longer
        #if self.cfg.cd_type == CDType.pcd:
        #    global gTemp
        #    tempold, gTemp = gTemp, 10
        #    for i in xrange(20):
        #        gTemp = gTemp/1.12
        #        self.pcdStep(layer,self.b1_tmp,self.b2_tmp,self.cfg.batchsize)
        #    gTemp = tempold
        #    ### run a bit longer with normal temperature
        #    for i in xrange(5):
        #        self.pcdStep(layer,self.b1_tmp,self.b2_tmp,self.cfg.batchsize)

        timestr = ""
        if "last_time_stamp" in self.__dict__:
            ts   = time.clock()
            td = ts - self.last_time_stamp
            if td > 0 and iter!=self.last_time_stamp_iter:
                timestr = " %2.4es/img ; %2.4e img/s"% ( td / (self.cfg.batchsize*(iter - self.last_time_stamp_iter)), (self.cfg.batchsize*(iter - self.last_time_stamp_iter))/td)

        err = self.getErr(layer,mbatch_provider.sampleset)
        self.reconstruction_error.append(err)
        n   = cp.norm2(self.weights[layer].mat.vec)
        print "Iter: ",iter, "Err: %02.06f |W|=%2.2f"%(err,n), timestr
        if self.cfg.save_every!=0 and iter % self.cfg.save_every == 0 and iter>0 :
            self.saveLayer(layer,self.cfg.workdir,"-%010d"%iter)
            self.saveOptions({"iter":iter},layer)
            self.saveOptions({"reconstruction":err},layer,"-%010d"%iter)

            # write pcd chain to images:
            #pchain=cp.pull(self.layers[layer].pchain)
            #for i,image in enumerate(pchain[:,:10].T):
                #plt.matshow(image.reshape((28,28)))
                #plt.savefig(os.path.join(self.cfg.workdir,"figure-pchain-%010d-chain%05d.png"%(iter,i)))
             

        self.err.append(err)
        self.last_time_stamp = time.clock()
        self.last_time_stamp_iter = iter

    def loadOptions(self,layer=0,postfix=""):
        fn= os.path.join(self.cfg.workdir,"info-%s%s.pickle"%(layer,postfix))
        # load pickle with statistics
        if os.path.exists(fn):
            with open(fn,"r") as f:
                return cPickle.load(f)
        return None

    def saveOptions(self,optionDict,layer=0,postfix=""):
        fn= os.path.join(self.cfg.workdir,"info-%s%s.pickle"%(layer,postfix))
        # load pickle with statistics
        if os.path.exists(fn):
            with open(fn,"r") as f:
                options=cPickle.load(f)
                for option,value in optionDict.items():
                    options[option]=value
        else:
            options=optionDict

        with open(fn,"wb") as f:
            cPickle.dump(options,f)


    def run(self, iterstart, itermax, mbatch_provider):
        """ Trains all levels of the RBM stack for itermax epochs. Lowest-level data comes from mbatch_provider. """
        self.mbp = mbatch_provider
        self.err = []
        mbatch_provider_orig=mbatch_provider
        for layer in xrange( self.cfg.num_layers-1 ):
            if layer >= self.cfg.continue_learning-1:
                try:
                    self.trainLayer(mbatch_provider,iterstart,itermax, layer)
                    print "Finished Layer ", layer
                except KeyboardInterrupt:
                    mbatch_provider.forgetOriginalData()
                    print "Stopping training of layer %d" % layer
                finally:
                    self.saveLayer(layer,self.cfg.workdir,"-pretrain")
            if layer < self.cfg.num_layers-2:
                mbatch_provider = self.getHiddenRep(layer,mbatch_provider)
                print "Got ", len(mbatch_provider.dataset), "batches"
        if self.cfg.dbm:
            print("Starting DBM training")
            try:
                self.trainDBM(mbatch_provider_orig,itermax)
            except KeyboardInterrupt:
                print("Done doing DBM training")
            finally:
                for layernum in xrange(len(self.weights)):
                    self.saveLayer(layernum,self.cfg.workdir,"-dbm")

    def load(self, prefix,load_type):
        """ Load saved weight matrices """
        if load_type==LoadType.latest:
            if os.path.exists(self.cfg.workdir + "/weights-0-finetune.npy"):
                load_type=LoadType.finetuning
            elif os.path.exists(self.cfg.workdir + "/weights-0-dbm.npy"):
                load_type=LoadType.dbm
            elif os.path.exists(self.cfg.workdir + "/weights-0-pretrain.npy"):
                load_type=LoadType.pretraining

        if load_type==LoadType.pretraining:
            postfix="pretrain"
        elif load_type==LoadType.dbm:
            postfix="dbm"
        elif load_type==LoadType.finetuning:
            postfix="finetune"

        for layer in xrange( self.cfg.num_layers-1 ):
            self.loadLayer(layer,prefix,postfix)

    def getHiddenRepDBM(self,mbatch_provider):
        """ Get hidden representation of the visible layer in all hidden layers of a DBM """
        rep_list=repList (len(self.layers),mbatch_provider.teacher)
        id=0
        while True:
            try:
                mbatch_provider.getMiniBatch(self.cfg.batchsize, self.layers[0].act, id=id)
            except MiniBatchProviderEmpty:
                return rep_list
            mbatch_provider.forgetOriginalData()
            uq = UpdateQ(len(self.layers))
            uq.push([1]) # must start w/ 0-th layer
            while uq.minupdates([0]) < 10:
                layernum = uq.pop(firstlayer=1)
                self.updateLayer(layernum,sample=False,blur=False)

            for i in xrange(1,len(self.layers)):
                rep_list.appendRep(i-1,cp.pull(self.layers[i].act))
                id +=1
        return rep_list


    def getHiddenRep(self, layer, mbatch_provider):
        """ Get hidden representation of a level in the RBM """
        id = 0
        mblist = []
        while True:
            try:
                mbatch_provider.getMiniBatch(self.cfg.batchsize, self.layers[layer].act, id=id)
            except MiniBatchProviderEmpty:
                return minibatch_provider.ListMiniBatchProvider(mblist)
            mbatch_provider.forgetOriginalData()
            self.upPass(layer,sample=False,blur=False)
            top_layer=self.layers[layer+1]
            max_pool(top_layer)
            mblist.append( cp.pull( top_layer.act) )
            id += 1

    def project_down(self):
        """ Project filters down to first (visible) layer """
        self.projection_results = dict()
        self.projection_results_lateral = dict()
        print "projecting down..."
        seqs = dict()

        for layernum in xrange(1,len(self.layers)):
            # project down from layer ``layernum''
            layer = self.layers[layernum]

            # turn on exactly one unit per layer
            cp.fill(layer.act_supersampled,0)
            if self.cfg.batchsize>layer.size:
                seqs[layernum] = np.array(xrange(0,layer.size))
            else:
                #seqs[layernum] = np.array(random.sample(xrange(layer.size), self.cfg.batchsize))
                seqs[layernum] = np.array(xrange(self.cfg.batchsize))

            # first pooling stuff:
            if self.cfg.maxpooltype == MaxPoolType.first:
                px=np.sqrt(layer.size/layer.maps)
                seqs[layernum]=layer.factor*(seqs[layernum]%px)  + layer.factor**2*px*(seqs[layernum]/int(px))

            for i in xrange(len(seqs[layernum])):
                layer.act_supersampled.set(int(seqs[layernum][i]),i,1)

            for i in reversed(xrange(1,layernum+1)):
                lower_layer = self.layers[i-1]
                self.weights[i-1].downPass(lower_layer,self.layers[i],sample=False,supersample=True)
                supersample(lower_layer)

            img = cp.pull( self.layers[0].act )
            img -= np.tile(img.mean(axis=1), (img.shape[1],1)).T
            img -= np.tile(img.mean(axis=0), (img.shape[0],1))
            self.projection_results[layernum] = img

        if self.cfg.srbmhid:
            layer=self.layers[1]
            print("Projecting lateral connections down")
            for i in xrange(len(seqs[1])):
                layer.act_supersampled.set(int(seqs[1][i]),i,1)
            tmp=get_copy(layer.act_supersampled)
            cp.prod(layer.act_supersampled,self.weights[0].latMat,tmp,'t','n')
            #pdb.set_trace()
            layer.nonlinearity()
            tmp.dealloc()
            self.weights[0].downPass(self.layers[0],self.layers[1],sample=False,supersample=True)
            lat = cp.pull( self.layers[0].act )
            #lat -= np.tile(lat.mean(axis=1), (lat.shape[1],1)).T
            #lat -= np.tile(lat.mean(axis=0), (lat.shape[0],1))
            self.projection_results_lateral[layernum] = lat



    def prepare_dbg(self,mbatch_provider,Npoint,nsteps,eval_start,save_callback):
        """ Prepare the data for visualization """
        print "Preparing data for visualization..."
        mbatch_provider.getMiniBatch(self.cfg.batchsize, self.layers[0].act)
        if  eval_start == EvalStartType.trainingset:
            print "Starting Eval from Trainingset"
            pass
        elif eval_start == EvalStartType.vnoise:
            print "Starting Eval from VNoise"
            cp.fill_rnd_uniform(self.layers[0].act)
            cp.apply_scalar_functor(self.layers[0].act,cp.scalar_functor.MULT,0.3)
        elif eval_start == EvalStartType.h1noise:
            print "Starting Eval from H1Noise"
            cp.fill_rnd_uniform(self.layers[1].act)
            cp.apply_scalar_functor(self.layers[1].act,cp.scalar_functor.MULT,0.3)
            self.downPass(1,sample=True)
        elif eval_start == EvalStartType.incomplete:
            print "Andy was to lazy to implement this yet"
            #self.layers[0].act.pull()
            #vis_=cp.create_numpy_from_mat_copy(self.layers[0].act)
            #vis=cp.create_mat_from_numpy_view("vis_hack",vis_) 
            #self.layers[0].act.dealloc()
            #self.layers[0].act=vis
            #save_vis=vis_[0:int(vis_.shape[0]/2),:].copy()
            #vis_[int(vis_.shape[0]/2):-1,:]=0
            #self.layers[0].act.push()
        self.dbg_datout    = []
        video              = self.cfg.video
        for layer_num,layer in enumerate(self.layers[0:-1]):
            self.upPass(layer_num, sample=False, blur=False)
            if layer_num+2 < len(self.layers):
                layer_above=self.layers[layer_num+1]
                max_pool(layer_above)

        if self.cfg.dbm:
            uq = UpdateQ(len(self.layers))
            uq.push([1]) # start with some layer in between
            step = 0
            while uq.minupdates([]) < nsteps:
                layernum = uq.pop(firstlayer=0)
                if video and layernum == 0:
                    self.updateLayer(layernum,sample=False,blur=True)
                    self.save_fantasy(step, Npoint,save_callback, self.layers[0].act)
                self.updateLayer(layernum,sample=True,blur=True)
                step+=1
            while uq.minupdates([]) < nsteps+2:
                layernum = uq.pop(firstlayer=0)
                self.updateLayer(layernum,sample=False,blur=False)
            self.updateLayer(0,sample=False,blur=False)
        else:
            num_meanfield=100
            for step in xrange(nsteps+num_meanfield):
                sample=not step>nsteps
                self.upPass(self.cfg.num_layers-2, sample=sample, blur=False)
                if video:
                    for lay_num in reversed(xrange(1,self.cfg.num_layers)):
                          self.downPass(lay_num, sample=False,supersample=True)
                    #if eval_start == EvalStartType.incomplete:
                        #self.rbms[0].vis.act.pull()
                        #vis_[0:int(vis_.shape[0]/2),:]=save_vis
                        #self.rbms[0].vis.act.push()
                    self.save_fantasy(step, Npoint,save_callback, self.layers[0].act)

                #if eval_start == EvalStartType.incomplete:
                    #self.layers[0].act.pull()
                    #vis_[0:int(vis_.shape[0]/2),:]=save_vis
                    #self.layers[0].act.push()
                #top_layer=self.layers[cfg.num_layers-1]
                #max_pool(top_layer)

                self.downPass(self.cfg.num_layers-1,sample=sample,supersample=False)
            for layer_num in reversed(xrange(1,self.cfg.num_layers)):
              self.downPass(layer_num, sample=False,supersample=True)
              layer=self.layers[layer_num-1]
              supersample(layer)
            #for bla in xrange(1):
            #    self.downPass(1,sample=False)
            #    self.upPass(0,sample=False,blur=False)
        # pass up again before we save fantasies -- assures that we see bottom-up activities!
        for layer_num,layer in enumerate(self.layers[0:-1]):
            self.upPass(layer_num, sample=False, blur=False)
            if layer_num+2 < len(self.layers):
                layer_above=self.layers[layer_num+1]
                max_pool(layer_above)
        #self.save_fantasy(nsteps+2,Npoint,save_callback, self.layers[0].pchain)
        self.save_fantasy(nsteps+1,Npoint,save_callback, self.layers[0].act)
        self.dbg_sampleset = mbatch_provider.sampleset_[:, 0:Npoint].T
        print "Pulling Layer-Activations..."
        self.act = {}
        self.act_info = {}
        for l in xrange(1, self.cfg.num_layers):
            L = self.layers[l]
            if L.act_supersampled != L.act:
                    self.act_info["%d-supers"%l] = dict(maps=L.maps,px=np.sqrt(L.size_supersampled/L.maps), py=np.sqrt(L.size_supersampled/L.maps))
                    self.act["%d-supers"%l] = cp.pull(L.act_supersampled)
            if l<self.cfg.num_layers-1:
                self.act_info["%d-subs"%l]   = dict(maps=L.maps,px=np.sqrt(L.size/L.maps), py=np.sqrt(L.size/L.maps))
                self.act["%d-subs"%l]   = cp.pull(L.act)
        if self.weights[0].mat.w < 800*6:
            print "Trying to pull W0..."
            try:
                self.W=cp.pull(self.weights[0].mat)
                if len(self.weights)>1:
                    self.W1=cp.pull(self.weights[1].mat)
            except MemoryError:
                print("weights too big!")
        if self.cfg.srbmhid:
            if self.weights[0].latMat.w < 800*3:
                print "Trying to pull I0..."
                try:
                    self.I=cp.pull(self.weights[0].latMat)
                except MemoryError:
                    print("lateral weights too big!")
        print "done."

    def save_fantasy(self,step,Npoint,cb,activations):
        #self.dbg_datout.append(cp.pull(self.layers[0].act)[0:self.layers[0].size,0:Npoint].T)
        cb(step,cp.pull(activations)[0:self.layers[0].size,0:Npoint].T)
    def saveAllLayers(self,postfix):
        for layernum in xrange(self.cfg.num_layers-1):
            self.saveLayer(layernum,self.cfg.workdir,postfix)


def initialize(cfg):
    cp.initCUDA( cfg.device )
    cp.initialize_mersenne_twister_seeds(cfg.seed)
