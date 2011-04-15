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

# debugging libs

from helper_functions import *
from helper_classes import *
import minibatch_provider

class WeightLayer(object):
    def __init__(self,layer1,layer2,cfg,layernum):
        self.mat=cp.dev_tensor_float_cm([layer1.size,layer2.size])
        cp.fill(self.mat,0)
        cp.add_rnd_normal(self.mat)
        fact = 1.0
        if layer2.unit_type == UnitType.binary or layer1.unit_type == UnitType.binary:
            # the 0.5 stems from the fact that our upper layer has activation 0.5 on average, not 0, if we use binary hidden units.
            fact = 0.5

        cp.apply_scalar_functor(self.mat,cp.scalar_functor.MULT,
                                fact/math.sqrt(max(layer1.size, layer2.size)))
        self.allocBias(layer1,layer2)
        self.num_params = self.mat.size + len(self.bias_lo) + len(self.bias_hi)

    def allocBias(self,layer1,layer2):
        self.bias_lo=cp.dev_tensor_float(layer1.size)
        self.bias_hi=cp.dev_tensor_float(layer2.size)
        cp.fill(self.bias_lo,0)
        cp.fill(self.bias_hi,0)
    def save(self,prefix,postfix):
        np.save(os.path.join(prefix,"weights-%s.npy"%postfix),self.mat.np)
        np.save(os.path.join(prefix,"bias-lo-%s.npy"%postfix),self.bias_lo.np)
        np.save(os.path.join(prefix,"bias-hi-%s.npy"%postfix),self.bias_hi.np)
    def load(self,prefix,postfix):
        fn = os.path.join(prefix,"weights-%s.npy"%postfix)
        if os.path.exists(fn):
            self.mat.dealloc()
            self.mat=cp.push(np.load(fn))
            self.bias_lo.dealloc()
            self.bias_hi.dealloc()
            self.bias_lo = cp.push(np.load(os.path.join(prefix,"bias-lo-%s.npy"%postfix)))
            self.bias_hi = cp.push(np.load(os.path.join(prefix,"bias-hi-%s.npy"%postfix)))
    def downPass(self,layer1,layer2,sample):
        cp.prod(layer1.act,self.mat,layer2.act,'n','n')
        layer1.postUpdateFromAbove(sample,bias=self.bias_lo)
    def upPass(self,layer1,layer2,sample):
        cp.prod(layer2.act,self.mat,layer1.act,'t','n')
        layer2.postUpdateFromBelow(sample,bias=self.bias_hi)
    def allocUpdateMatrix(self):
        self.w_tmp =cp.dev_tensor_float_cm(self.mat.shape)
        cp.fill(self.w_tmp,0)
        self.blo_tmp=cp.dev_tensor_float(len(self.bias_lo))
        self.bhi_tmp=cp.dev_tensor_float(len(self.bias_hi))
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
        cp.learn_step_weight_decay(self.mat,self.w_tmp,learnrate,cost) # W  += learnrate(dW - cost*W)
        cp.learn_step_weight_decay(self.bias_lo,self.blo_tmp,learnrate,cost) # W  += learnrate(dW - cost*W)
        cp.learn_step_weight_decay(self.bias_hi,self.bhi_tmp,learnrate,cost) # W  += learnrate(dW - cost*W)
    def updateGradientNeg(self,layer1,layer2,batchsize):
        cp.prod(self.w_tmp,layer1.act,layer2.act,'n','t',-1./batchsize,1./batchsize)
        cp.reduce_to_col(self.blo_tmp,layer1.act,cp.reduce_functor.ADD,-1./batchsize,1./batchsize)
        cp.reduce_to_col(self.bhi_tmp,layer2.act,cp.reduce_functor.ADD,-1./batchsize,1./batchsize)
    def updateGradientPos(self,layer1,layer2):
        cp.prod(self.w_tmp,layer1.act,layer2.act,'n','t')
        cp.reduce_to_col(self.blo_tmp,layer1.act)
        cp.reduce_to_col(self.bhi_tmp,layer2.act)


class NodeLayer(object):
    def __init__(self, size, batchsize, unit_type,sigma):
        self.size = size
        self.bsize = batchsize
        self.unit_type = unit_type
        self.sigma=sigma
        self.alloc()
    def sample(self):
        if self.unit_type == UnitType.gaussian or self.unit_type == UnitType.cont:
            cp.add_rnd_normal(self.act)
        elif self.unit_type == UnitType.binary:
            cp.rnd_binarize(self.act)
    def alloc(self):
        self.act = cp.dev_tensor_float_cm([self.size,self.bsize])
        cp.fill(self.act,0)
        return self
    def dealloc(self):
        self.act.dealloc()
    def nonlinearity(self):
        if not self.unit_type == UnitType.gaussian:
            cp.apply_scalar_functor(self.act,cp.scalar_functor.SIGM)
    def allocPChain(self):
        self.pchain=cp.dev_tensor_float_cm([self.size, self.bsize])
        cp.fill(self.pchain,0)
    def deallocPChain(self):
        if not "pchain" in self.__dict__:
            return
        self.pchain.dealloc()
    def switchToPChain(self):
        self.org= self.act
        self.act= self.pchain
    def switchToOrg(self):
        self.pchain= self.act
        self.act   = self.org
    def postUpdateFromAbove(self,sample,bias):
        cp.matrix_plus_col(self.act,bias)
        self.nonlinearity()
        if sample:
            self.sample()
    def postUpdateFromBelow(self,sample,bias):
        cp.matrix_plus_col(self.act,bias)
        self.nonlinearity()
        if sample:
            self.sample()


class RBMStack:
    """ Represents a stack of Restricted Boltzmann Machines """
    def __init__(self, cfg):
        """ Creates a stack of RBMs with cfg.num_layers levels. Each Layer is initialized using cfg."""
        self.layers = []
        self.weights= []
        self.cfg = cfg;
        self.states = {}

        for layer in xrange(self.cfg.num_layers):
            self.layers.append(NodeLayer(self.cfg.l_size[layer], self.cfg.batchsize,self.cfg.utype[layer],cfg.sigma))
            if self.cfg.cd_type == CDType.pcd:
                self.layers[layer].allocPChain()
        else:
            weight_type=WeightLayer

        for layer in xrange(self.cfg.num_layers-1):
            self.weights.append(weight_type(self.layers[layer],self.layers[layer+1],self.cfg, layernum=layer))

        self.reconstruction_error = []
        self.matrequests_new = {}
        self.matrequests     = {}
        self.pulled = {}
    def getErr(self,layernum,orig_data):
        cp.apply_binary_functor(self.layers[layernum].act,orig_data,cp.binary_functor.SUBTRACT)
        sqerr = cp.norm2(self.layers[layernum].act)**2
        return sqerr/((self.layers[layernum].size)*self.cfg.batchsize)
    def saveLayer(self,layernum,prefix,postfix):
        self.weights[layernum].save(prefix,str(layernum)+postfix)
    def loadLayer(self,layernum,prefix,postfix):
        self.weights[layernum].load(prefix,str(layernum)+'-'+postfix)
    def upPass(self, layernum,sample=True):
        self.weights[layernum].upPass(self.layers[layernum],self.layers[layernum+1],sample)
    def downPass(self, layernum,sample=True):
        self.weights[layernum-1].downPass(self.layers[layernum-1],self.layers[layernum],sample=sample)
    def updateLayer(self,layernum,sample=True):
        L = self.layers[layernum]
        if layernum==0:
            self.downPass(layernum+1,sample=sample)
        if layernum==len(self.layers)-1:
            self.upPass(layernum-1,sample)
        if layernum<len(self.layers)-1 and layernum>0: 
            hi = self.layers[layernum+1]
            lo = self.layers[layernum-1]
            wlo = self.weights[layernum-1]
            whi = self.weights[layernum]

            cp.prod(L.act,whi.mat,hi.act,'n','n')
            cp.matrix_plus_col(L.act,whi.bias_lo)

            tmp = get_copy(L.act)
            cp.prod(L.act,wlo.mat,lo.act,'t','n')
            cp.matrix_plus_col(L.act,wlo.bias_hi)

            # add parts from above/below
            cp.apply_binary_functor(L.act,tmp,cp.binary_functor.AXPBY,0.5,0.5)
            tmp.dealloc()

            L.nonlinearity()
            if sample:
                L.sample()

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
            self.upPass(layernum,sample=True)
            self.downPass(layernum+1,sample=True)
            self.weights[layernum].updateGradientNeg(self.layers[layernum],self.layers[layernum+1],self.cfg.batchsize)
        finally:
            self.layers[layernum].switchToOrg()
            self.layers[layernum+1].switchToOrg()

    def cdnStep(self,layernum,batchsize):
        for step in xrange(self.cfg.ksteps):
            self.upPass(layernum,sample=True)
            self.downPass(layernum+1,sample=True)
        self.upPass(layernum,sample=True)
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
                for layernum in xrange(len(self.weights)):
                    self.upPass(layernum,sample=False)
                uq = UpdateQ(len(self.layers))
                uq.push([1]) # must start w/ 0-th layer
                while uq.minupdates([0]) < self.cfg.dbm_minupdates:
                    layernum = uq.pop(firstlayer=1)
                    self.updateLayer(layernum,sample=False)
                for layernum in xrange(len(self.weights)):
                    self.weights[layernum].updateGradientPos(self.layers[layernum],self.layers[layernum+1])

                ### output stuff
                if iter != 0 and (iter%100) == 0:
                    self.downPass(1,sample=False)
                    err=self.getErr(0,mbatch_provider.sampleset)
                    self.reconstruction_error.append(err)
                    print "Iter: ",iter, "Err: %02.06f"%err, "|W|: %02.06f"%cp.norm2(self.weights[0].mat)
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
                    self.updateLayer(layernum,sample=True)

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
                self.upPass(layernum,sample=False)
                self.weights[layernum].updateGradientPos(self.layers[layernum],self.layers[layernum+1])
                ### negative phase
                if self.cfg.cd_type == CDType.cdn:
                    self.cdnStep(layernum,self.cfg.batchsize)
                elif self.cfg.cd_type == CDType.pcd:
                    self.pcdStep(layernum,self.cfg.batchsize)
                ### update weights and biases
                self.weights[layernum].updateStep(learnrate,self.cfg.cost)
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
        
        self.upPass(layer,sample=False);     
        self.downPass(layer+1,sample=False);     
        for l in reversed(xrange(1,layer+1)):
            self.downPass(l,sample=False)


        timestr = ""
        if "last_time_stamp" in self.__dict__:
            ts   = time.clock()
            td = ts - self.last_time_stamp
            if td > 0 and iter!=self.last_time_stamp_iter:
                timestr = " %2.4es/img ; %2.4e img/s"% ( td / (self.cfg.batchsize*(iter - self.last_time_stamp_iter)), (self.cfg.batchsize*(iter - self.last_time_stamp_iter))/td)
        err = self.getErr(layer,mbatch_provider.sampleset)
        self.reconstruction_error.append(err)
        n   = cp.norm2(self.weights[layer].mat)
        print "Iter: ",iter, "Err: %02.06f |W|=%2.2f"%(err,n), timestr
        if self.cfg.save_every!=0 and iter % self.cfg.save_every == 0 and iter>0 :
            self.saveLayer(layer,self.cfg.workdir,"-%010d"%iter)
            self.saveOptions({"iter":iter},layer)
            self.saveOptions({"reconstruction":err},layer,"-%010d"%iter)

            # write pcd chain to images:
            #pchain=self.layers[layer].pchain.np
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
                self.updateLayer(layernum,sample=False)

            for i in xrange(1,len(self.layers)):
                rep_list.appendRep(i-1,self.layers[i].act.np)
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
            self.upPass(layer,sample=False)
            top_layer=self.layers[layer+1]
            mblist.append(top_layer.act.np)
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
            cp.fill(layer.act,0)
            if self.cfg.batchsize>layer.size:
                seqs[layernum] = np.array(xrange(0,layer.size))
            else:
                #seqs[layernum] = np.array(random.sample(xrange(layer.size), self.cfg.batchsize))
                seqs[layernum] = np.array(xrange(self.cfg.batchsize))

            for i in xrange(len(seqs[layernum])):
                layer.act.set(int(seqs[layernum][i]),i,1)

            for i in reversed(xrange(1,layernum+1)):
                lower_layer = self.layers[i-1]
                self.weights[i-1].downPass(lower_layer,self.layers[i],sample=False)

            img = self.layers[0].act.np
            img -= np.tile(img.mean(axis=1), (img.shape[1],1)).T
            img -= np.tile(img.mean(axis=0), (img.shape[0],1))
            self.projection_results[layernum] = img

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
            #self.layers[0].act.np()
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
            self.upPass(layer_num, sample=False)
            if layer_num+2 < len(self.layers):
                layer_above=self.layers[layer_num+1]

        if self.cfg.dbm:
            uq = UpdateQ(len(self.layers))
            uq.push([1]) # start with some layer in between
            step = 0
            while uq.minupdates([]) < nsteps:
                layernum = uq.pop(firstlayer=0)
                if video and layernum == 0:
                    self.updateLayer(layernum,sample=False)
                    self.save_fantasy(step, Npoint,save_callback, self.layers[0].act)
                self.updateLayer(layernum,sample=True)
                step+=1
            while uq.minupdates([]) < nsteps+2:
                layernum = uq.pop(firstlayer=0)
                self.updateLayer(layernum,sample=False)
            self.updateLayer(0,sample=False)
        else:
            num_meanfield=100
            for step in xrange(nsteps+num_meanfield):
                sample=not step>nsteps
                self.upPass(self.cfg.num_layers-2, sample=sample)
                if video:
                    for lay_num in reversed(xrange(1,self.cfg.num_layers)):
                          self.downPass(lay_num, sample=False)
                    self.save_fantasy(step, Npoint,save_callback, self.layers[0].act)

                self.downPass(self.cfg.num_layers-1,sample=sample)
            for layer_num in reversed(xrange(1,self.cfg.num_layers)):
              self.downPass(layer_num, sample=False)
              layer=self.layers[layer_num-1]
            #for bla in xrange(1):
            #    self.downPass(1,sample=False)
            #    self.upPass(0,sample=False)
        # pass up again before we save fantasies -- assures that we see bottom-up activities!
        for layer_num,layer in enumerate(self.layers[0:-1]):
            self.upPass(layer_num, sample=False)
            if layer_num+2 < len(self.layers):
                layer_above=self.layers[layer_num+1]
        self.save_fantasy(nsteps+1,Npoint,save_callback, self.layers[0].act)
        self.dbg_sampleset = mbatch_provider.sampleset_[:, 0:Npoint].T
        print "Pulling Layer-Activations..."
        self.act = {}
        self.act_info = {}
        for l in xrange(1, self.cfg.num_layers):
            L = self.layers[l]
            if l<self.cfg.num_layers-1:
                self.act_info["%d-subs"%l]   = dict(px=np.sqrt(L.size), py=np.sqrt(L.size))
                self.act["%d-subs"%l]   = L.act.np
        if self.weights[0].mat.w < 800*6:
            print "Trying to pull W0..."
            try:
                self.W=self.weights[0].mat.np
                if len(self.weights)>1:
                    self.W1=self.weights[1].mat.np
            except MemoryError:
                print("weights too big!")
        print "done."

    def save_fantasy(self,step,Npoint,cb,activations):
        #self.dbg_datout.append(self.layers[0].act.np[0:self.layers[0].size,0:Npoint].T)
        cb(step,activations.np[0:self.layers[0].size,0:Npoint].T)
    def saveAllLayers(self,postfix):
        for layernum in xrange(self.cfg.num_layers-1):
            self.saveLayer(layernum,self.cfg.workdir,postfix)


def initialize(cfg):
    cp.initCUDA( cfg.device )
    cp.initialize_mersenne_twister_seeds(cfg.seed)
