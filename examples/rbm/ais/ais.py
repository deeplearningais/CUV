import time
import sys

from time import strftime
import os
import numpy as np
import cPickle

sys.path.append(".")
sys.path.append("..")

import cuv_python as cp
from sensibleconfig import Config, Option
from helper_classes import Dataset, UnitType
from datasets import MNISTData, ShifterData, BarsAndStripesData


def sigm(x):
    return 1/(1+np.exp(-x))

def sample(x,unit_type):

    if unit_type=="gaussian":
        y = (x + np.random.normal(0,1,x.shape)).astype('float32')
    else:
        noise = (np.random.uniform(0,1,x.shape)).astype('float32')
        y = (noise < x).astype("float32")
    return y

def rect(x):
    i=x<87
    x[i]=np.log(1+np.exp(x[i]))
    return x


class ais(object):
    def __init__(self,cfg,dataset) :
        self.cfg=cfg
        self.dataset = dataset
        self.data=self.dataset.data.astype(np.float32)
        # normalize data between 0 and 1
        self.data=(self.data-self.data.min())/self.data.max()
        self.v_org = self.data.astype("float64")

    def initialize_everything(self):
        ### initialize matrices for cuda ###
        self.r_ =np.zeros(self.cfg.chains).astype('float32').copy('F')
        self.w = cp.dev_tensor_float_cm(self.w_.copy("F"))


        ### generate basemodel
        softened = (self.data.mean(axis=1) + 0.1)
        self.baserate_bias_= (np.log(softened) - np.log(1-softened)).astype('float32').copy('F')
        self.baserate_bias_.shape=(self.w.shape[0],1)


        ## start chains
        self.v_ = np.tile(sigm(self.baserate_bias_),(1,self.cfg.chains))
        self.v_ = sample(self.v_,self.cfg['utype']).astype('float32').copy('F')
        self.v = cp.dev_tensor_float_cm(self.v_.copy("F"))
        self.h = cp.dev_tensor_float_cm([self.num_hids,self.cfg.chains])

        self.baserate_bias = cp.dev_tensor_float_cm(self.baserate_bias_.copy("F"))
        self.r = cp.dev_tensor_float_cm(np.vstack(self.r_).copy("F"))
        cp.initialize_mersenne_twister_seeds(int(time.time()*1000) % 100000)

    def get_partition_function(self):
        tmp = cp.dev_tensor_float_cm([self.cfg.chains, 1])
        tmp2 = cp.dev_tensor_float_cm([self.num_hids,self.cfg.chains])
        #steps = 14500
        #steps = 1000
        steps = self.cfg.steps
        #beta=0.001
        beta = 1.0/steps
        beta_old=0
        for step in xrange(steps):
            self.p_k(beta_old,tmp,tmp2,lambda x: cp.apply_binary_functor(self.r,x,cp.binary_functor.SUBTRACT))
            self.p_k(beta,tmp,tmp2,lambda x: cp.apply_binary_functor(self.r,x,cp.binary_functor.ADD))
            self.sample_markov_chains(beta,step)
            ### sample v_i
            ### increase beta
            beta_old = beta
            #if step<500:
                #beta += 0.001
            #elif step < 4500:
                #beta += 0.0001
            #else :
                #beta += 0.00001
            beta += 1.0/steps
            #if step % 100 == 0:
                #self.r_=self.r.np
                #v_=self.v.np
                #h_=self.h.np
                #print "v: %f"%v_.mean()
                #print "h: %f"%h_.mean()
                #print "r: %f"%self.r_.mean()
                #sys.stdout.write('.')
                #sys.stdout.flush()

        ### multiply r by partition function of baseline rbm
        self.r_=self.r.np
        self.partition_baserate = (np.log(1+np.exp(self.baserate_bias_))).sum()+self.num_hids*np.log(2)
        self.r_ += self.partition_baserate
        tmp.dealloc()
        tmp2.dealloc()

    def run(self):

        fn= os.path.join(self.cfg.path,"info-0-%s.pickle"%self.cfg.postfix)

        # load pickle with statistics
        if os.path.exists(fn):
            with open(fn,"r") as f:
                stats=cPickle.load(f)
        else:
            # create new empty statistics
            stats=dict()

        steps=self.cfg.steps

        # if pickle already contains data, do nothing
        if "ais_z_%d"%steps in stats and 'ais_p_nats_%d'%steps in stats and 'ais_z_std_%d'%steps in stats and not stats['ais_p_nats_%d'%steps] in [np.inf,np.nan,-np.inf]:
            print("file %s already contains data"%fn)
        else:
            start_time = time.clock()
            self.load_weights(self.cfg.path)
            self.initialize_everything()

            self.get_partition_function()
            Z_dev=self.r_.std()
            Z_mean = self.r_.mean()
            print "\npartition function: " + str(Z_mean),
            print "standarddeviation: " + str(Z_dev)
            p_v = self.get_data_likelihood()
            print "dataloglikelihood  mean: " + str((p_v).mean())+ " std-deviation: " + str(p_v.std())

            end_time = time.clock()
            stats['ais_z_%d'%steps]=Z_mean
            stats['ais_p_nats_%d'%steps]=p_v.mean()
            stats['ais_z_std_%d'%steps]=Z_dev

            stats['ais_dev'] = self.cfg.device
            stats['basename'] = self.cfg.path
            stats['bindex'] = self.cfg.postfix
            stats['ais_host'] = os.uname()[1]
            stats['ais_time'] = strftime("%Y-%m-%d %H:%M:%S")
            stats['ais_duration'] = end_time - start_time

            with open(fn,"wb") as f:
                cPickle.dump(stats,f)

    def get_data_likelihood(self):
        ### calculate data loglikelihood
        p_v= (rect(np.dot(self.w_.T,self.v_org)+self.bias_hi.np).sum(axis=0) +np.dot(self.bias_lo.np.T,self.v_org))-self.r_.mean()
        print "datalikelihood under baserate model: ", (np.dot(self.baserate_bias_.T,self.v_org) - self.partition_baserate).mean()
        return p_v

    def load_weights(self,path):
        print "loading weights from ",path
        if not os.path.exists(os.path.join(path,"weights-0-%s.npy"%self.cfg.postfix)):
            print "Could not open weights."
            sys.exit(1)
        self.w_ =np.load(os.path.join(path,"weights-0-%s.npy"%self.cfg['postfix']))
        self.bias_lo = cp.dev_tensor_float((np.load(os.path.join(path,"bias-lo-0-%s.npy"%self.cfg.postfix))).reshape(-1,1))
        self.bias_hi = cp.dev_tensor_float((np.load(os.path.join(path,"bias-hi-0-%s.npy"%self.cfg.postfix))).reshape(-1,1))
        self.w=cp.dev_tensor_float_cm(self.w_.copy("F"))
        self.num_vis=self.w_.shape[0]
        self.num_hids=self.w_.shape[1]
        print "Number of hidden units: ",self.num_hids

    def p_k(self,beta,tmp,tmp2,collect):
        cp.prod(tmp,self.v,self.baserate_bias,'t','n')
        cp.apply_scalar_functor(tmp,cp.scalar_functor.MULT,(1-beta))
        collect(tmp)
        cp.prod(tmp2,self.w,self.v,'t','n')
        cp.matrix_plus_col(tmp2,self.bias_hi)

        cp.apply_scalar_functor(tmp2,cp.scalar_functor.MULT,beta)

        # RECT computes log(1+exp(x))
        cp.apply_scalar_functor(tmp2,cp.scalar_functor.RECT,1)

        cp.reduce_to_row(tmp.T,tmp2,cp.reduce_functor.ADD) # tmp.T is an evil hack. it makes tmp into row major, which doesn't change anything since it's a vector any way. But vectors are always assumed to be row major.
        collect(tmp)
        cp.prod(tmp,self.v,self.bias_lo.T,'t','n')
        cp.apply_scalar_functor(tmp,cp.scalar_functor.MULT,beta)
        collect(tmp)

    def sample_markov_chains(self,beta,step):
        cp.prod(self.h,self.w,self.v,'t','n')
        cp.matrix_plus_col(self.h,self.bias_hi)
        cp.apply_scalar_functor(self.h,cp.scalar_functor.MULT,beta)
        cp.apply_scalar_functor(self.h,cp.scalar_functor.SIGM)
        cp.rnd_binarize(self.h)
        cp.prod(self.v,self.w,self.h,'n','n')
        cp.matrix_plus_col(self.v,self.bias_lo)
        cp.apply_scalar_functor(self.v,cp.scalar_functor.MULT,beta)
        cp.apply_scalar_functor(self.baserate_bias,cp.scalar_functor.MULT,1-beta)

        cp.matrix_plus_col(self.v,self.baserate_bias)
        cp.apply_scalar_functor(self.baserate_bias,cp.scalar_functor.MULT,1.0/(1-beta))
        cp.apply_scalar_functor(self.v,cp.scalar_functor.SIGM)
        #if step % 100 == 0:
           #plt.figure(1)
           #self.v_=self.v.np
           #showthis = self.v_.copy()
           #plt.matshow(showthis[:,0].reshape((28,28)))
           #plt.draw()
           #if not os.path.exists("/tmp/%s"%os.getlogin()):
               #os.mkdir("/tmp/%s"%os.getlogin())
           #plt.savefig("/tmp/%s/chain_%05d.png"%(os.getlogin(),step))
        cp.rnd_binarize(self.v)


if __name__=='__main__':

    loc = locals()
    options = [
        Option('device',    'GPU device to use [%default]', 0, short_name='d', converter=int),
        Option('chains',     'number of markov chains [%default]', 512, short_name='c', converter=int),
        Option('steps', 'number of annealing steps (don\'t change this at the moment [%default]', 145000,  short_name='s', converter=int),
        Option('postfix',     'index/postfix of file [%default]', "pretrain", short_name='i', converter=str),
        Option('utype',  'type of visible units (cont/gaussian/binary) [%default]', 'binary.binary', short_name='V', converter=lambda x: [eval("UnitType."+y, loc) for y in x.split(".")]),
        Option('path',    'Path to weights [%default]', '.', short_name='p', converter=str),
]

    cfg = Config(options, usage = "usage: %prog [options]")
    try:
        sys.path.insert(0,os.path.join(os.getenv('HOME'),'bin'))
        import optcomplete
        optcomplete.autocomplete(cfg._parser)
    except ImportError:
        pass

    cfg.grab_from_argv(sys.argv)
    cfg.batchsize=-1 # not used, only compatibility with dataset classes


    #load pickle with rbm cfg
    fn = os.path.join(cfg.path, "info-0.pickle")
    with open(fn,"r") as f:
        rbmcfg=cPickle.load(f)
    cfg.dataset   = rbmcfg['dataset']

    if cfg.dataset==Dataset.mnist:
        dataset = MNISTData(cfg,"/home/local/datasets/MNIST")
    elif cfg.dataset==Dataset.shifter:
        dataset = ShifterData(cfg,"/home/local/datasets/")
    elif cfg.dataset==Dataset.bars_and_stripes:
        dataset = BarsAndStripesData(cfg,"/home/local/datasets/")

    cp.initCUDA(cfg.device)
    my_ais =ais(cfg,dataset)
    my_ais.run()
    cp.exitCUDA()
