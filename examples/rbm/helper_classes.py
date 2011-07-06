import numpy as np
import operator, random

class CDType:            pcd, cdn, mpfl               = range(3)
class WeightUpdateMeth:  rprop, backprop              = range(2)
class UnitType:          cont, gaussian, binary       = range(3)
class LearnRateSchedule: static, linear, exponential, divide  = range(4)
class EvalStartType:     trainingset, h1noise, vnoise = range(3)
class LoadType:          none,pretraining, dbm, finetuning,latest = range(5)
class Dataset:           image_patches, mnist, mnist_padded, mnist_test,caltech,caltech_color,caltech_big,mnist_twice,mnist_trans,one_minus_mnist,shifter,bars_and_stripes = range(12)

class UpdateQ(object):
    def __init__(self,numlayers):
        self.q = []
        self.d = {}
        self.numlayers = numlayers
        # initialize d[i] to 0
        map(operator.setitem, [self.d]*numlayers, range(numlayers), [0]*numlayers)

    def push(self, layers):
        self.q.extend(layers)

    def minupdates(self,excl=[]):
        d  = self.d
        d2 = [d[x] for x in self.d.keys() if x not in excl]
        return min(d2)

    def pop(self,firstlayer=0):
        x = random.choice(self.q)
        self.q = [k for k in self.q if k != x]
        if x   > firstlayer:       self.q.append(x-1)
        if x+1 < self.numlayers:   self.q.append(x+1)
        self.d[x] += 1
        return x

    def num_updates(self):
        return np.sum(np.array(self.d.values()))

class repList(object):
    def __init__(self,num_layers,labels):
        self.num_layers=num_layers
        self.layers = []
        for i in xrange(self.num_layers-1):
            self.layers.append([])
        self.labels=labels
    def appendRep(self,layer_num,rep):
        if self.layers[layer_num]!=[]:
            self.layers[layer_num]=np.hstack((self.layers[layer_num],rep))
        else:
            self.layers[layer_num]=rep
class MaxPoolType: first,plain, soft = range(3)

