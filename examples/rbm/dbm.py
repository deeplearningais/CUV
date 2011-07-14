import sys
import numpy as np

import cuv_python as cp

from helper_classes import CDType, UpdateQ, repList, EvalStartType
from minibatch_provider import MiniBatchProviderEmpty
from base import RBMStack

class DBM(RBMStack):
    def __init__(self, cfg):
        super(DBM, self).__init__(cfg)

    def run(self,isterstart, itermax, mbatch_provider):
        super(DBM, self).run(isterstart, itermax, mbatch_provider)
        mbatch_provider_orig=mbatch_provider
        print("Starting DBM training")
        try:
            self.trainDBM(mbatch_provider_orig,itermax)
        except KeyboardInterrupt:
            print("Done doing DBM training")
        finally:
            for layernum in xrange(len(self.weights)):
                self.saveLayer(layernum,self.cfg.workdir,"-dbm")

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
        self.dbg_datout    = []
        video              = self.cfg.video
        for layer_num,layer in enumerate(self.layers[0:-1]):
            self.upPass(layer_num, sample=False)

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
        # pass up again before we save fantasies -- assures that we see bottom-up activities!
        for layer_num,layer in enumerate(self.layers[0:-1]):
            self.upPass(layer_num, sample=False)
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
        if self.weights[0].mat.shape[0] < 800*6:
            print "Trying to pull W0..."
            try:
                self.W=self.weights[0].mat.np
                if len(self.weights)>1:
                    self.W1=self.weights[1].mat.np
            except MemoryError:
                print("weights too big!")
        print "done."


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
                    print "Iter: ",iter, "Err: %02.06f"%err, "|W|: %02.06f"%cp.norm2(self.weights[0].mat)
                    print self.cfg.workdir,
                    if self.cfg.save_every!=0 and iter % self.cfg.save_every == 0 and iter>0 :
                        for layernum in xrange(len(self.weights)):
                            self.saveLayer(layernum,self.cfg.workdir,"-%010d"%iter)

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
