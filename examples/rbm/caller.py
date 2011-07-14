#!/usr/bin/python
import sys
import time
import os
import numpy as np
import cPickle as pickle
from datasets import MNISTData, MNISTOneMinusData, MNISTPadded, MNISTTwiceData, MNISTTestData, \
        CaltechData, ShifterData, BarsAndStripesData, ImagePatchesData
from sensibleconfig import Config, Option

import base as pyrbm
from dbm import DBM
import minibatch_provider
import cuv_python as cp
from helper_functions import visualize_rows, make_img_name
from helper_classes import Dataset

def make_img(x, px,py,maps,mbs,isweight=False):
    #if not isweight and cfg.utype[0] != pyrbm.UnitType.binary:
    if (cfg.utype[0]!=pyrbm.UnitType.binary) and not isweight:
        if maps > 1:
            out= x.reshape((px,py,maps))
        else:
            out= 255-x.reshape(px,py)
    else:
        out = x.reshape((maps*px,py))
    return out

np.random.seed(36487)

loc = locals()
options = [
    Option('config_file','Use defaults from this file', None, short_name='c', converter=str),
    Option('headless',     'whether to exit without opening windows in the end [%default]', 0,                   short_name='H', converter=int),
    Option('device',     'GPU device to use [%default]', 0,                   short_name='d', converter=int),
    Option('seed',     'Seed to use [%default]', int(time.time()*1000) % 100000,                  converter=int),
    Option('ksteps',     'number of CD/PCD steps [%default]', 1,                 short_name='k', converter=int),
    Option('batchsize', 'number of items in a minibatch - default is dataset-dependend', -1,  short_name='b', converter=int),
    Option('weight_updates',     'number of weight_updates [%default]', 300000,                short_name='e', converter=int),
    Option('l_size',     'size of layers w/o bias [%default]', "512", short_name='n', converter=lambda x: [int(y) for y in x.split(".")]),
    Option('img_size',  'input is n x n [%default]', 28, converter=int),
    Option('utype',  'type of visible units (cont/gaussian/binary) [%default]', 'binary.binary', short_name='V', converter=lambda x: [eval("pyrbm.UnitType."+y, loc) for y in x.split(".")]),
    Option('cd_type',    'variant of Contrastive Divergence (pcd,cdn,mpfl) [%default]', 'pcd',     short_name='p', converter=lambda x: eval("pyrbm.CDType."+x, loc)),
    Option('learnrate_sched', 'learnrate schedule (static/linear/exponential/divide) [%default]', 'static', short_name='s', converter=lambda x: eval("pyrbm.LearnRateSchedule."+x, loc)),
    Option('sigma', 'noise added to gaussian/cont unit [%default]', 1.0, converter=float),

    Option('momentum',   'momentum learnrate of weights [%default]', 0.0, converter=float),
    Option('learnrate',  'learnrate of weights [%default]', 0.010, converter=float),
    Option('cost',       'cost of weights (=weight decay) [%default]', 0.0000, converter=float),

    Option('eval_start', 'start evaluation using trainingset/vnoise/h1noise [%default]', 'vnoise', converter=lambda x: eval("pyrbm.EvalStartType."+x, loc)),
    Option('eval_steps', 'number of sampling steps for visualization [%default]', 100, converter=int),
    Option('video', 'whether to write [eval_steps] frames to files [%default]', False, converter=int),

    Option('workdir', 'directory-prefix of data to load/save [%default]', ".", short_name="w", converter=str),
    Option('load', 'which weights to load (none,pretraining,dbm,finetuning,latest) [%default]', "none", converter=lambda x:eval("pyrbm.LoadType."+x,loc)),
    Option('project_down', 'load weights and project down to visible layer [%default]', False, converter=int),
    Option('dbm', 'whether to use unsupervised finetuning', False, converter=int),
    Option('dbm_minupdates', 'how many updates per layer before moving to next minibatch [%default]', 5, converter=int),
    Option('save_every', 'Save weights every ... iters [%default]', 0, converter=int),
    Option('dataset', 'whether to use mnist or image patches or caltech [%default]', 'mnist', converter=lambda x: eval("Dataset."+x, loc)),
    Option('continue_learning', 'whether to use loaded weights to continue learning [%default]', 0, converter=int),
    Option('gethidrep', 'whether to dump hidden rep after loading [%default]', False, converter=int),

    Option('pretrain', 'whether to dump matrices and configs without doing anything [%default]', True, converter=int),
    Option('finetune', 'whether to do fine tuning after initializing/learning weights [%default]', False, converter=int),
    Option('finetune_epochs', 'epochs for fine-tuning [%default]', 100, converter=int),
    Option('finetune_rprop', 'whether to use rprop for fine-tuning [%default]', 1, converter=int),
    Option('finetune_batch_size', 'size of minibatches for fine-tuning [%default]', 1000, converter=int),
    Option('finetune_softmax', 'whether to use softmax for fine-tuning [%default]', 1, converter=int),
    Option('finetune_learnrate', 'learnrate for finetuning [%default]', 0.01, converter=float),
    Option('finetune_cost', 'weight decay for finetuning [%default]', 0.001, converter=float),
    Option('finetune_momentum', 'momentum for finetuning when backprop used [%default]', 0.5, converter=float),
    Option('finetune_onlylast', 'epochs where only not-pretrained upper layer is updated [%default]', 0, converter=int),
    Option('finetune_online_learning', 'online learning [%default]', '0', converter=int),
]

cfg = Config(options, usage = "usage: %prog [options]")

try:
    sys.path.insert(0,os.path.join(os.getenv('HOME'),'bin'))
    import optcomplete
    optcomplete.autocomplete(cfg._parser)
except ImportError:
    pass


cfg.grab_from_argv(sys.argv)    # to fetch the config file name
try:
    os.mkdir(cfg.workdir)
except:pass
cfg.config_file = os.path.join(cfg.workdir, "pyrbm.ini")

print "Loading configuration from ", cfg.config_file

if os.path.exists(cfg.config_file):
    cfg.grab_from_file(cfg.config_file)
cfg.grab_from_argv(sys.argv)    # overwrite using cmdline
cfg.num_layers=len(cfg.l_size)+1
dic = cfg.to_dict()
cfgk= dic.keys(); cfgk.sort()
if (cfg.project_down or cfg.gethidrep or cfg.continue_learning) and cfg.load==pyrbm.LoadType.none:
    cfg.load=pyrbm.LoadType.latest
print "#",
print "".join(map(lambda x:(("%20s:"%x)+str(dic[x])),cfgk))
print ""

if cfg.dbm==1 and cfg.num_layers==2:
    print "dbm training not possible with only one layer"
    sys.exit(1)

cfg.maps_bottom=1

if cfg.dataset==Dataset.mnist:
    dataset = MNISTData(cfg,"/home/local/datasets/MNIST")
elif cfg.dataset==Dataset.one_minus_mnist:
    dataset = MNISTOneMinusData(cfg,"/home/local/datasets/MNIST")
elif cfg.dataset==Dataset.mnist_padded:
    dataset = MNISTPadded(cfg,"/home/local/datasets/MNIST")
elif cfg.dataset==Dataset.mnist_twice:
    dataset = MNISTTwiceData(cfg,"/home/local/datasets/MNIST")
elif cfg.dataset==Dataset.mnist_test:
    dataset = MNISTTestData(cfg,"/home/local/datasets/MNIST")
elif cfg.dataset==Dataset.image_patches:
    dataset = ImagePatchesData(cfg,os.getenv("HOME"))
elif cfg.dataset==Dataset.caltech or cfg.dataset==Dataset.caltech_big:
    dataset = CaltechData(cfg,"/home/local/datasets/batches","gray",0)
elif cfg.dataset==Dataset.caltech_color:
    dataset = CaltechData(cfg,"/home/local/datasets/batches","color",0)
elif cfg.dataset==Dataset.shifter:
    dataset = ShifterData(cfg,"/home/local/datasets/")
    if cfg.batchsize!=768:
        print("WARNING: Batchsize %d != 768 but 768 recommended for Shifter dataset."%cfg.batchsize)
elif cfg.dataset==Dataset.bars_and_stripes:
    dataset = BarsAndStripesData(cfg,"/home/local/datasets/")
    if cfg.batchsize!=32:
        print("WARNING: Batchsize %d != 32 but 32 recommended for Bars and Stripes dataset."%cfg.batchsize)

#### set correct visible layer size
cfg.l_size.insert(0,cfg.px*cfg.py*cfg.maps_bottom)

assert (cfg.pretrain or not cfg.continue_learning)
### append node types
while len(cfg.utype) < len(cfg.l_size):
    cfg.utype.append(pyrbm.UnitType.binary)

print "Shuffling data..."
if not cfg.gethidrep:
    dataset.shuffle()

print "Initializing RBM..."
pyrbm.initialize(cfg)
print "ready."

if cfg.dbm:
    rbmstack = DBM(cfg)
else:
    rbmstack = pyrbm.RBMStack(cfg)

rbmstack.saveOptions(cfg.get_serialization_obj())

mbp = minibatch_provider.MNISTMiniBatchProvider(dataset.data, dataset.teacher)

print "Calculating statistics for minibatch..."
mbs = minibatch_provider.MiniBatchStatistics(mbp, rbmstack.layers[0].act)
if cfg.utype[0] == pyrbm.UnitType.gaussian:
   mbp.norm = lambda x: mbs.normalize_zmuv(x)
else:
   mbp.norm = lambda x: mbs.normalize_255(x)
if "test_data" in dataset.__dict__:
   mbp_test = minibatch_provider.MNISTMiniBatchProvider(dataset.test_data, dataset.test_teacher) 
   mbp_test.norm = mbp.norm
mbp.mbs = mbs # allows visualization of mean, range, etc 
print "done."

if cfg.load!=pyrbm.LoadType.none:
    print "Loading RBMStack"
    rbmstack.load(cfg.workdir,cfg.load)
    if cfg.project_down:
        rbmstack.project_down()
    if cfg.gethidrep:
        print "Saving rbm hidden representations...",
        x = dict([[v,k] for k,v in Dataset.__dict__.items()])
        descr = x[cfg.dataset]
        f=open(os.path.join(cfg.workdir,"%s.pickle"%descr),'w')
        if cfg.dbm==1:
            pickle.dump(rbmstack.getHiddenRepDBM(mbp),f,-1)
        else :
            mbp2 = mbp
            hid_rep_list=pyrbm.repList(len(rbmstack.layers),mbp2.teacher)
            for i in xrange(len(rbmstack.layers)-1):
                print str(i), "...",
                mbp2 = rbmstack.getHiddenRep(i, mbp2)
                hid_rep_list.appendRep(i,np.hstack(mbp2.dataset))
            pickle.dump(hid_rep_list,f,-1)
        f.close()

        if "test_data" in dataset.__dict__:
            print "Saving rbm hidden reps for test set"
            descr = descr+"_test"
            f=open(os.path.join(cfg.workdir,"%s.pickle"%descr),'w')
            if cfg.dbm==1:
                pickle.dump(rbmstack.getHiddenRepDBM(mbp_test),f,-1)
            else :
                mbp2=mbp_test
                hid_rep_list=pyrbm.repList(len(rbmstack.layers),mbp2.teacher)
                for i in xrange(len(rbmstack.layers)-1):
                    print str(i), "...",
                    mbp2 = rbmstack.getHiddenRep(i, mbp2)
                    hid_rep_list.appendRep(i,np.hstack(mbp2.dataset))
                pickle.dump(hid_rep_list,f,-1)
            f.close()
            print "done."
        cfg.continue_learning=0
        cfg.finetune=0

    if cfg.continue_learning:
        assert(not cfg.load==pyrbm.LoadType.finetuning)
        options=rbmstack.loadOptions()
        if options:
            iter_start=options['iter']+1
        else:
            iter_start=0

        rbmstack.run(iter_start,cfg.weight_updates, mbp)

elif cfg.pretrain:
    rbmstack.run(0,cfg.weight_updates, mbp)

if cfg.finetune:

    # clean up RBM parts which are not needed anymore
    map(lambda x:x.deallocPChain(), rbmstack.layers)
    map(lambda x:x.dealloc(),       rbmstack.layers)

    weights = map(lambda x: x.mat,    rbmstack.weights)
    biases  = map(lambda x: x.bias_hi, rbmstack.weights)
    from mlp import MLP
    pymlp = MLP(cfg, weights,biases)

    pymlp.preEpochHook = lambda mlp,epoch: epoch%10==0 and mlp.runMLP(mbp_test, cfg.test_batchsize,epoch)
    try:
        pymlp.train(mbp,cfg.finetune_epochs, cfg.finetune_batch_size, cfg.finetune_rprop)
    except KeyboardInterrupt:
        pass
    map(lambda x:x.alloc(),     rbmstack.layers)
    map(lambda x:x.allocPChain(), rbmstack.layers)
    rbmstack.saveAllLayers("-finetune")
    pymlp.saveLastLayer()

if cfg.headless:
   cp.exitCUDA()
   sys.exit(0)
PLT_NUM=1
import matplotlib.pyplot as plt

#### calculate maps_bottom into py. yeah it's a dirty hack, i know
px=cfg.px
py=cfg.py

if "projection_results" in rbmstack.__dict__:
    for layernum in rbmstack.projection_results.keys():
        filters = rbmstack.projection_results[layernum].T
        print "Saving projections from layer %d (%d x %d)" % (layernum,filters.shape[0],filters.shape[1])
        img_name = make_img_name("filter_layer%d.png"%(layernum))
        visualize_rows(PLT_NUM,filters,range(20), lambda x:make_img(x,px,py,cfg.maps_bottom,True), title='Projection W Layer %d'%layernum, use_imshow=cfg.maps_bottom>1, save=True, save_filename=img_name, normalize=False)
        PLT_NUM+=1
    sys.exit(0)

def save_fantasy(step,drec,cfg,pnum):
    plt.close('all')
    img_name = make_img_name("fantasy_%05d.png"%step)
    visualize_rows(pnum,drec,range(20), lambda x:make_img(x,cfg.px,cfg.py,cfg.maps_bottom,mbs,True), title='Fantasy/Reconstruction '+str(step+1), use_imshow=cfg.maps_bottom>1, save=True, save_filename=img_name, normalize=True)

cb = lambda step, drec: save_fantasy(step,drec,cfg,PLT_NUM)
rbmstack.prepare_dbg(mbp,30,cfg.eval_steps,cfg.eval_start,cb)
PLT_NUM += 1
fan=np.array(rbmstack.dbg_datout)

if "W" in rbmstack.__dict__ :
    W_old=rbmstack.W.copy()
    w=rbmstack.W.T
    start = 0* px*px
    end   = start + px*px
    visualize_rows(PLT_NUM,w,range(20), lambda x:make_img(x,px,py,cfg.maps_bottom,mbs,True), title="Weights/Filters", use_imshow=cfg.maps_bottom>1,cb=False,normalize=True)
    PLT_NUM+=2
else:
    print "Can't load weights"

# originals
visualize_rows(PLT_NUM,rbmstack.dbg_sampleset,range(20), lambda x:make_img(x,px,py,[4,1][cfg.maps_bottom==1],mbs,False), title='Originals', use_imshow=cfg.maps_bottom>1)
PLT_NUM+=1

cp.exitCUDA()

try:
    plt.show()
except KeyboardInterrupt:
    plt.close('all')

