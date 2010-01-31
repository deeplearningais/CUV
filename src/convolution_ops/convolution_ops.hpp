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
void convolve2(dev_dense_matrix<V,M,I>& dst,
		   dev_dense_matrix<V,M,I>& img,
		   dev_dense_matrix<V,M,I>& filter,
		   int numFilters);

template<class V, class M, class I>
void convolve2(host_dense_matrix<V,M,I>& dst,
		   host_dense_matrix<V,M,I>& img,
		   host_dense_matrix<V,M,I>& filter,
		   int numFilters);

template<class V, class M, class I>
void convolve3(dev_dense_matrix<V,M,I>& dst,
		   dev_dense_matrix<V,M,I>& img,
		   dev_dense_matrix<V,M,I>& filter);

template<class V, class M, class I>
void convolve3(host_dense_matrix<V,M,I>& dst,
		   host_dense_matrix<V,M,I>& img,
		   host_dense_matrix<V,M,I>& filter);

template<class V, class M, class I>
void gridToMatrix(dev_dense_matrix<V,M,I>& img,
		   dev_dense_matrix<V,M,I>& grid,
		   int poolSize);
template<class V, class M, class I>
void matrixToGrid(dev_dense_matrix<V,M,I>& grid,
		   dev_dense_matrix<V,M,I>& img,
		   int poolSize);
template<class V, class M, class I>
void localMaximum(dev_dense_matrix<V,M,I>& dst,
		   dev_dense_matrix<V,M,I>& img,
		   int poolSize);

template<class V, class M, class I>
void localMaximum(host_dense_matrix<V,M,I>& dst,
		   host_dense_matrix<V,M,I>& img,
		   int poolSize);

template<class V, class M, class I>
void supersample(host_dense_matrix<V,M,I>& dst,
		host_dense_matrix<V,M,I>& img,
		int factor);

template<class V, class M, class I>
void supersample(dev_dense_matrix<V,M,I>& dst,
		dev_dense_matrix<V,M,I>& img,
		int factor);

template<class V, class M, class I>
void reorder(dev_dense_matrix<V,M,I>& A,
		   int blockLength);

template<class V, class M, class I>
void reorder(host_dense_matrix<V,M,I>& A,
		   int blockLength);


template<class V, class M, class I>
void superToMax(host_dense_matrix<V,M,I>& bigError,
		host_dense_matrix<V,M,I>& smallError,
		host_dense_matrix<V,M,I>& bigImg,
		host_dense_matrix<V,M,I>& smallImg,
		int factor);

template<class V, class M, class I>
void superToMax(dev_dense_matrix<V,M,I>& bigError,
		dev_dense_matrix<V,M,I>& smallError,
		dev_dense_matrix<V,M,I>& bigImg,
		dev_dense_matrix<V,M,I>& smallImg,
		int factor);

template<class V, class M, class I>
void copyInto(dev_dense_matrix<V,M,I>& dst,
		   dev_dense_matrix<V,M,I>& img,
		   int padding);

template<class V, class M, class I>
void copyInto(host_dense_matrix<V,M,I>& dst,
		   host_dense_matrix<V,M,I>& img,
		   int padding);

}


#endif /* __CONVOLUTION_OPS_HPP__ */
