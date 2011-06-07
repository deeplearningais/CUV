import sys
import numpy as np
import cuv_python as cp
class MiniBatchProviderEmpty(IndexError):pass
class MiniBatchProvider:
    def getMiniBatch(self): raise NotImplementedError
    def setMiniBatch(self, mb, dst_layer):
        self.sampleset_ = mb
        self.sampleset  = cp.dev_tensor_float_cm(self.sampleset_.astype('float32').copy('F'))
        cp.copy(dst_layer,self.sampleset)

    def forgetOriginalData(self):
        if "sampleset" in self.__dict__:
            self.sampleset.dealloc()
            self.sampleset_ = None

class MNISTMiniBatchProvider(MiniBatchProvider):
    def __init__(self, mnist_data, teacher = None):
        self.dataset = mnist_data
        self.pos = 0
        self.teacher = teacher
        if teacher != None:
            assert mnist_data.shape[1] == teacher.shape[1]
    def __len__(self):
        return self.dataset.shape[1]

    def getMiniBatch(self, samplesize, dst_layer, id=None, return_teacher=False):
        if id == None:
            #id = np.random.randint(0,len(self.dataset)-samplesize)
            id = self.pos
            self.pos = self.pos + samplesize
            self.pos = self.pos % self.dataset.shape[1]
            if self.dataset.shape[1] < self.pos+samplesize-1:
                self.pos = 0
            id = self.pos
        else:
            id = id*samplesize
        if self.dataset.shape[1] < id+samplesize:
            raise MiniBatchProviderEmpty
        self.setMiniBatch(self.dataset[:,id:id+samplesize], dst_layer)
        if return_teacher:
            return cp.dev_tensor_float_cm(self.teacher[:,id:id+samplesize].astype('float32').copy('F'))

class MiniBatchStatistics:
    def update_stats(self,batch):
        vmin  = cp.dev_tensor_float(batch.shape[0])
        vmax  = cp.dev_tensor_float(batch.shape[0])
        mean  = cp.dev_tensor_float(batch.shape[0])
        mean2 = cp.dev_tensor_float(batch.shape[0])
        map(lambda x: cp.fill(x,0), [mean,mean2])
        cp.reduce_to_col(mean,batch)
        cp.reduce_to_col(mean2,batch,cp.reduce_functor.ADD_SQUARED)
        cp.reduce_to_col(vmin,batch,cp.reduce_functor.MIN)
        cp.reduce_to_col(vmax,batch,cp.reduce_functor.MAX)
        if "N" in self.__dict__:
            self.N += batch.shape[1]
            cp.apply_binary_functor(self.mean, mean, cp.binary_functor.ADD)
            cp.apply_binary_functor(self.mean2,mean2,cp.binary_functor.ADD)
            cp.apply_binary_functor(self.min,vmin,cp.binary_functor.MIN)
            cp.apply_binary_functor(self.max,vmin,cp.binary_functor.MAX)
            mean.dealloc()
            mean2.dealloc()
            vmin.dealloc()
            vmax.dealloc()
        else:
            self.N     = batch.shape[1]
            self.mean  = mean
            self.mean2 = mean2
            self.min   = vmin
            self.max   = vmax

    def finalize_stats(self):
        """ use N, mean and mean2 to generate data for normalization """

        # mean := (mean/N)^2
        cp.apply_scalar_functor(self.mean,cp.scalar_functor.MULT,1./self.N)
        sqmean = self.mean.copy()
        cp.apply_scalar_functor(sqmean, cp.scalar_functor.SQUARE)

        # mean2 -= mean2/n - squared_mean
        cp.apply_scalar_functor(self.mean2,cp.scalar_functor.MULT,1./self.N)
        cp.apply_binary_functor(self.mean2,sqmean,cp.binary_functor.SUBTRACT)

        # std is sqrt of difference
        cp.apply_scalar_functor(self.mean2,cp.scalar_functor.ADD,0.01) # numerical stability
        cp.apply_scalar_functor(self.mean2,cp.scalar_functor.SQRT)
        self.std = self.mean2
        sqmean.dealloc()

        # negate mean (so we can add it to normalize a matrix)
        cp.apply_scalar_functor(self.mean,cp.scalar_functor.MULT,-1.)
        self.negative_mean = self.mean

        # calculate range
        cp.apply_binary_functor(self.max, self.min, cp.binary_functor.SUBTRACT)
        cp.apply_scalar_functor(self.max, cp.scalar_functor.MAX, 1.)
        self.range = self.max
        # calculate negative min
        cp.apply_scalar_functor(self.range,cp.scalar_functor.ADD,0.01) # numerical stability
        cp.apply_scalar_functor(self.min,cp.scalar_functor.MULT,-1.)
        self.negative_min = self.min

        assert not cp.has_nan(self.negative_mean)
        assert not cp.has_inf(self.negative_mean)
        assert not cp.has_nan(self.std)
        assert not cp.has_inf(self.std)
        assert not cp.has_nan(self.negative_min)
        assert not cp.has_inf(self.range)

    def normalize_zmuv(self,batch):
        """ Zero Mean, Unit Variance based on recorded statistics """
        cp.matrix_plus_col(batch,self.negative_mean)
        cp.matrix_divide_col(batch,self.std)

    def normalize_255(self,batch):
        """ normalize by subtracting min and dividing by range"""
        cp.apply_scalar_functor(batch,cp.scalar_functor.DIV, 255.)

    def normalize_minmax(self,batch):
        """ normalize by subtracting min and dividing by range"""
        cp.matrix_plus_col(batch,self.negative_min)
        cp.matrix_divide_col(batch,self.range)

    #def normalize_min2(self,batch):
    #    """ normalize by subtracting min and dividing by range"""
    #    cp.matrix_plus_col(batch,self.negative_min)
    #    cp.apply_scalar_functor(batch,cp.scalar_functor.DIV,2.)

    def __init__(self, mbp, act):
        """ generate statistics for the data in a minibatch-provider """
        try:
            sid = 0
            while True:
                sys.stdout.write('.')
                sys.stdout.flush()
                mbp.getMiniBatch(act.shape[1],act, sid)
                mbp.forgetOriginalData()
                self.update_stats(act)
                sid += 1
        except MiniBatchProviderEmpty:
            self.finalize_stats()


class MovedMiniBatchProvider(MNISTMiniBatchProvider):
    def __init__(self, data, src_size, dst_size,src_num_maps, teacher=None, maxmov=5, noise_std=0):
        self.dataset = data
        self.pos = 0
        self.src_size = src_size
        self.dst_size = dst_size
        self.src_num_maps = src_num_maps
        self.norm = None
        self.teacher = teacher
        self.maxmov = maxmov
        self.noise_std = noise_std
    def set_translation_max(self, tm): self.maxmov = tm
    def set_noise_std(self,n):         self.noise_std = n
    def setMiniBatch(self, mb, dst_layer):
        self.sampleset_ = mb
        self.sampleset  = cp.dev_tensor_uc_cm(self.sampleset_.copy('F'))
        shift = np.random.randint(-self.maxmov,self.maxmov+1, 2).astype('int32')
        cp.image_move(dst_layer,self.sampleset,self.src_size, self.dst_size, self.src_num_maps, shift[0],shift[1])
        if self.noise_std != 0:
            cp.add_rnd_normal(dst_layer,self.noise_std)
        if self.norm:
            self.norm(dst_layer)
        self.sampleset.dealloc()
        self.sampleset = dst_layer.copy()

class ListMiniBatchProvider(MiniBatchProvider):
    def __init__(self, list):
        self.dataset = list
        self.pos = 0
    def __len__(self):
        return len(self.dataset) * self.dataset[0].shape[1]
    def getMiniBatch(self, samplesize, dst_layer, id=None):
        if id == None:
            #id = np.random.randint(0,len(self.dataset))
            id = self.pos
            self.pos = self.pos + 1
            self.pos = self.pos % len(self.dataset)
        if id >= len(self.dataset):
            raise MiniBatchProviderEmpty
        self.setMiniBatch(self.dataset[id], dst_layer)
