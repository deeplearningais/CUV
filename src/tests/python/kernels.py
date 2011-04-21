#!/usr/bin/env python

import numpy as np
import cuv_python as cp
from nose.tools import *


def test_pairwise_euclidean_dist():
    from scipy.spatial.distance import cdist
    x = np.random.uniform(0,1,(20,10))
    y = np.random.uniform(0,1,(30,10))
    x_ = cp.dev_tensor_float(x)
    y_ = cp.dev_tensor_float(y)
    dists = cp.dev_tensor_float([x_.shape[0],y_.shape[0]])
    cp.pairwise_distance_l2(dists,x_,y_)
    numpy_dist = cdist(x,y)
    ok_(np.linalg.norm(numpy_dist-dists.np)<1e-3)

def test_pairwise_euclidean_dist_cm():
    from scipy.spatial.distance import cdist
    x = np.random.uniform(0,1,(20,10))
    y = np.random.uniform(0,1,(30,10))
    x_ = cp.dev_tensor_float_cm(x.copy('F'))
    y_ = cp.dev_tensor_float_cm(y.copy('F'))
    dists = cp.dev_tensor_float_cm([x_.shape[0],y_.shape[0]])
    cp.pairwise_distance_l2(dists,x_,y_)
    numpy_dist = cdist(x,y)
    ok_(np.linalg.norm(numpy_dist-dists.np)<1e-3)
