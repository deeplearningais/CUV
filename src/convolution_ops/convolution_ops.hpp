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
void sample_multinomial(dev_dense_matrix<V,M,I>& grid);

template<class V, class M, class I>
void prob_max_pooling(dev_dense_matrix<V,M,I>& grid, int poolSize, bool sample);
template<class V, class M, class I>
void prob_max_pooling(dev_vector<V,I>& sums, dev_dense_matrix<V,M,I>& grid, int poolSize, bool sample);

template<class V, class M, class I>
void grid_to_matrix(dev_dense_matrix<V,M,I>& mat,
		   dev_dense_matrix<V,M,I>& grid,       
		   int poolSize);
template<class V, class M, class I>
void matrix_to_grid(dev_dense_matrix<V,M,I>& grid,
		   dev_dense_matrix<V,M,I>& mat,
		   int poolSize);

template<class V, class M, class I>
void supersample(host_dense_matrix<V,M,I>& dst,
		host_dense_matrix<V,M,I>& img,
		int factor,
		host_dense_matrix<int,row_major>* indices = NULL);

template<class V, class M, class I>
void supersample(dev_dense_matrix<V,M,I>& dst,
		dev_dense_matrix<V,M,I>& img,
		int factor,
		dev_dense_matrix<int,row_major>* indices = NULL);

template<class V, class M, class I>
void reorder(dev_dense_matrix<V,M,I>& A,
		   int blockLength);

template<class V, class M, class I>
void reorder(host_dense_matrix<V,M,I>& A,
		   int blockLength);


template<class V, class M, class I>
void super_to_max(host_dense_matrix<V,M,I>& dst,
		host_dense_matrix<V,M,I>& img,
		int poolSize,
		int overlap = 0,
		host_dense_matrix<int,row_major>* indices = NULL);


template<class V, class M, class I>
void super_to_max(dev_dense_matrix<V,M,I>& dst,
		dev_dense_matrix<V,M,I>& img,
		int poolSize,
		int overlap = 0,
		dev_dense_matrix<int,row_major>* indices = NULL);

template<class V, class M, class I>
void copy_into(dev_dense_matrix<V,M,I>& dst,
		   dev_dense_matrix<V,M,I>& img,
		   int padding);

template<class V, class M, class I>
void copy_into(host_dense_matrix<V,M,I>& dst,
		   host_dense_matrix<V,M,I>& img,
		   int padding);

template<class V, class M, class I>
void max_pooling(host_dense_matrix<V,M,I>& dst,
		host_dense_matrix<V,M,I>& img,
		unsigned int poolSize,
		unsigned int overlap = 0,
		host_dense_matrix<int,row_major>* indices = NULL,
		host_dense_matrix<V,M,I>* filter = NULL);

template<class V, class M, class I>
void max_pooling(dev_dense_matrix<V,M,I>& dst,
		dev_dense_matrix<V,M,I>& img,
		unsigned int poolSize,
		unsigned int overlap = 0,
		dev_dense_matrix<int,row_major>* indices = NULL,
		dev_dense_matrix<V,M,I>* filter = NULL);

template<class V, class M, class I>
void strip_padding(dev_dense_matrix<V,M,I>& dst,
				   dev_dense_matrix<V,M,I>& img,
				   unsigned int padding);

template<class V, class M, class I>
void strip_padding(host_dense_matrix<V,M,I>& dst,
				   host_dense_matrix<V,M,I>& img,
				   unsigned int padding);

template<class V, class M, class I>
void row_ncopy(dev_dense_matrix<V,M,I>& dst,
			   dev_vector<V,I>& row,
			   unsigned int n);
}


#endif /* __CONVOLUTION_OPS_HPP__ */
