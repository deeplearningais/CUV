#ifndef __CONVOLUTION_OPS_HPP__
#define __CONVOLUTION_OPS_HPP__

#include <stdio.h>

#include <vector_ops/vector_ops.hpp>
#include <basics/dev_dense_matrix.hpp>
#include <basics/host_dense_matrix.hpp>

namespace cuv{

/*
 * Wrappers for Alex' CUDA convolution functions
 */

template<class V, class M, class I>
void convolve(dev_dense_matrix<V,M,I>& dst,
		   dev_dense_matrix<V,M,I>& img,
		   dev_dense_matrix<V,M,I>& filter);

template<class V, class M, class I>
void convolve(host_dense_matrix<V,M,I>& dst,
		   host_dense_matrix<V,M,I>& img,
		   host_dense_matrix<V,M,I>& filter);

template<class V, class M, class I>
void localMaximum(dev_dense_matrix<V,M,I>& dst,
		   dev_dense_matrix<V,M,I>& img,
		   int poolSize);

template<class V, class M, class I>
void localMaximum(host_dense_matrix<V,M,I>& dst,
		   host_dense_matrix<V,M,I>& img,
		   int poolSize);

}


#endif /* __CONVOLUTION_OPS_HPP__ */
