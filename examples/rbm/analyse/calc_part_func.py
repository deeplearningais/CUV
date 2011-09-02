#!/usr/bin/python
import warnings
warnings.filterwarnings("ignore")
import cuv_python as cp
from progressbar import ProgressBar, Percentage, Bar, RotatingMarker, ETA
import numpy as np
import cPickle
import math
import time
import getopt
import sys
import os
from time import strftime

sys.path.append(".")
sys.path.append("..")
from datasets import MNISTData, ShifterData, BarsAndStripesData
from helper_classes import Dataset
import minibatch_provider

class PartitionFunction(object):
    def __init__(self, weight, bv,bh):
        self.weight = cp.dev_tensor_float_cm(weight)
        self.bv     = cp.dev_tensor_float(bv)
        self.bh     = cp.dev_tensor_float(bh)

    def partialsumV(self,actv, acth, row):
        """
         sums out hidden variables for given v
          exp( log(exp(bh + actv*W)+1).sum(axis=0) + (v*bv).sum(axis=0) )
        """
        # acth = bv + actv*W
        cp.prod(acth,self.weight,actv,'t','n')
        cp.matrix_plus_col(acth,self.bh)

        # acth = log(exp(acth)+1)
        cp.apply_scalar_functor(acth,cp.scalar_functor.RECT,1.0)

        # row = actv.sum(axis=0)
        cp.reduce_to_row(row,acth,cp.reduce_functor.ADD)

        # row += h*bh
        cp.matrix_times_col(actv,self.bv)
        cp.reduce_to_row(row,actv,cp.reduce_functor.ADD,1.0,1.0)

        # exp(row)
        m=row.np.astype("float64")

        return math.fsum(m.flatten())/actv.shape[1]

    def partialsum(self,acth, actv, row):
        """
        sums out visible variables for given hidden variables
          exp( log(exp(bv + acth*W)+1).sum(axis=0) + (h*bh).sum(axis=0) )
        """
        # actv = bv + acth*W
        cp.prod(actv,self.weight,acth,'n','n')
        cp.matrix_plus_col(actv,self.bv)

        # actv = log(exp(actv)+1)
        cp.apply_scalar_functor(actv,cp.scalar_functor.RECT,1.0)

        # row = actv.sum(axis=0)
        cp.reduce_to_row(row,actv,cp.reduce_functor.ADD)

        # row += h*bh
        cp.matrix_times_col(acth,self.bh)
        cp.reduce_to_row(row,acth,cp.reduce_functor.ADD,1.0,1.0)
        #cp.prod(row,self.bv,actv,'t','n',1.0,1.0)

        # exp(row)
        m=row.np.astype("float64")

        return math.fsum(np.exp(m).flatten())

    def numerator(self,mbp,batchsize):
        sid = 0
        actv = cp.dev_tensor_float_cm([self.weight.shape[0],batchsize])
        acth = cp.dev_tensor_float_cm([self.weight.shape[1],batchsize])
        row  = cp.dev_tensor_float([batchsize])
        cp.fill(acth,0.0)
        cp.fill(actv,0.0)
        cp.fill(row,0)
        print "Numerator: ",
        L    = []
        try:
            while True:
                mbp.getMiniBatch(batchsize,actv,sid)
                mbp.forgetOriginalData()
                sid  += 1
                L.append(self.partialsumV(actv,acth,row))
                sys.stdout.write('.')
                sys.stdout.flush()

        except minibatch_provider.MiniBatchProviderEmpty:
            print "done."
            pass
        for m in [actv,acth,row]: m.dealloc()
        return math.fsum(L) / (len(L))

    def denominator(self,batchsize):
        acth = cp.dev_tensor_float_cm([self.weight.shape[1], batchsize])
        actv = cp.dev_tensor_float_cm([self.weight.shape[0], batchsize])
        row  = cp.dev_tensor_float([batchsize])
        cp.fill(acth, 0.0)
        cp.fill(actv, 0.0)
        cp.fill(row, 0.0)
        n    = acth.shape[0]
        nmax = 2**n
        if nmax%batchsize != 0:
            print "Error: 2**n=%d must be dividable by batchsize=%d!"%(nmax,batchsize)
            sys.exit(1)
        L    = []
        widgets = ["Denominator: ", Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=nmax)
        for i in xrange(0,nmax,acth.shape[1]):
            cp.set_binary_sequence(acth,i)
            L.append(self.partialsum(acth,actv,row))
            if (i/acth.shape[1]) % 100 == 0:
                pbar.update(i)
        pbar.finish()
        for m in [actv,acth,row]: m.dealloc()
        return math.fsum(L)

def get_mbp(cfg):
    if cfg.dataset == Dataset.mnist:
        dataset = MNISTData(cfg,"/home/local/datasets/MNIST")
        mbp = minibatch_provider.MNISTMiniBatchProvider(dataset.data)
        act = cp.dev_tensor_float_cm([cfg.px*cfg.py, cfg.batchsize])
        mbs = minibatch_provider.MiniBatchStatistics(mbp,act)
        mbp.norm = lambda x: mbs.normalize_255(x)
        mbp.mbs = mbs # allows visualization of mean, range, etc
    elif cfg.dataset == Dataset.shifter:
        dataset = ShifterData(cfg,"/home/local/datasets")
        mbp = minibatch_provider.MNISTMiniBatchProvider(dataset.data)
    elif cfg.dataset == Dataset.bars_and_stripes:
        dataset = BarsAndStripesData(cfg,"/home/local/datasets")
        mbp = minibatch_provider.MNISTMiniBatchProvider(dataset.data)
    else:
        raise NotImplementedError()
    return mbp

def read_data(basename,idx):
    bv = os.path.join(os.path.join(basename, "bias-lo-0-%s.npy"%idx))
    bh = os.path.join(os.path.join(basename, "bias-hi-0-%s.npy"%idx))
    W  = os.path.join(os.path.join(basename, "weights-0-%s.npy"%idx))
    if not os.path.exists(bv):
        print "Could not open bv"
        sys.exit(1)
    if not os.path.exists(bh):
        print "Could not open bh"
        sys.exit(1)
    if not os.path.exists(W):
        print "Could not open W"
        sys.exit(1)
    bv = np.load(bv)
    bh = np.load(bh)
    W  = np.load(W)
    return bv, bh, W

def usage():
    print "Usage: %s --basename <basename> [--idx <index> ] [ --verbose ] [ --help ] [ --force-overwrite ] [ --device <devicenum> ]"%sys.argv[0]

class Cfg (object):pass

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:b:i:vf", ["help","device=", "basename=", "idx=","verbose","force-overwrite"])
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    cfg           = Cfg()
    cfg.verbose   = False
    #cfg.batchsize = 1024
    cfg.batchsize = 256
    cfg.device    = 0
    cfg.overwrite = False
    for o, a in opts:
        if o == "-v":
            cfg.verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-f", "--force-overwrite"):
            cfg.overwrite = True
        elif o in ("-d", "--device"):
            cfg.device = int(a)
        elif o in ("-b", "--basename"):
            cfg.basename = a
        elif o in ("-i", "--idx"):
            cfg.idx = a
        else:
            assert False, "unhandled option"
    if not hasattr(cfg,"basename"):
        usage()
        sys.exit()
    if not hasattr(cfg,"idx"):
        cfg.idx="pretrain"
    fn = os.path.join(cfg.basename, "info-0-%s.pickle"%cfg.idx)
    if not cfg.overwrite and os.path.exists(fn):
        with open(fn, "r") as f:
            stats = cPickle.load(f)
            if "nats" in stats:
                print "pickle already has a value for partition function in it, exiting."
                sys.exit(0)
    cp.initCUDA(cfg.device)

    #load pickle with rbm cfg
    fn = os.path.join(cfg.basename, "info-0.pickle")
    with open(fn,"r") as f:
        rbmcfg=cPickle.load(f)
    cfg.dataset   = rbmcfg['dataset']

    start_time = time.clock()
    bv,bh,W = read_data(cfg.basename, cfg.idx)
    mbp = get_mbp(cfg)
    pf = PartitionFunction(W,bv,bh)
    print "\nCalculating Probability of Data."

    if cfg.dataset == Dataset.mnist:
        num = pf.numerator(mbp,1000)
    elif cfg.dataset == Dataset.shifter:
        num = pf.numerator(mbp,768)
    elif cfg.dataset == Dataset.bars_and_stripes:
        num = pf.numerator(mbp,32)
    else:
        raise NotImplementedError()

    print "Numerator  : %2.10e nats" % num
    den = pf.denominator(cfg.batchsize)
    print "Denominator: %2.10e nats" % np.log(den)
    print "Resulting P: %2.10e nats" % (num - np.log(den))
    end_time = time.clock()

    fn = os.path.join(cfg.basename, "info-0-%s.pickle"%cfg.idx)
    if os.path.exists(fn):
        with open(fn, "r") as f:
            stats = cPickle.load(f)
    else:
        stats = {}

    stats['weights2']     = np.sum(W**2)
    stats['nats'    ]     = num-np.log(den)
    stats['nats_num']     = num
    stats['nats_den']     = np.log(den)
    stats['dev'     ]     = cfg.device
    stats['basename']     = cfg.basename
    stats['bindex'  ]     = cfg.idx
    stats['cpf_host']     = os.uname()[1]
    stats['cpf_duration'] = end_time - start_time
    stats['cpf_time'    ] = strftime("%Y-%m-%d %H:%M:%S")

    with open(fn,"wb") as f:
        cPickle.dump(stats,f)

    cp.exitCUDA()

if __name__ == "__main__":
    main()
