#ifndef __CONVOLUTION_OPS_HPP__
#define __CONVOLUTION_OPS_HPP__

#include <stdio.h>

#include <vector_ops.hpp>
#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>

//#include <cuv_general.hpp>
//#include <vector_ops.hpp>
//#include <dev_dense_matrix.hpp>
//#include <host_dense_matrix.hpp>
//#include <vector_ops/rprop.hpp>
//#include <convert/convert.hpp>

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

}


#endif /* __CONVOLUTION_OPS_HPP__ */
