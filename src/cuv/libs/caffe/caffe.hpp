//  Licensed under the BSD 2-Clause license.
//  CUV/src/cuv/libs/caffe/LICENSE

#ifndef CAFFE_OPS_HPP_
#define CAFFE_OPS_HPP_

#include <cuv.hpp>

namespace cuv {
/*
 * Wrappers for caffe functions
 */

/** @defgroup caffe_ops Caffe operations
 * @{
 */

namespace caffe {

	void empty_kernel_call();

	void create_cuda_stream(cudaStream_t &stream);

	void destroy_cuda_stream(cudaStream_t &stream);

	void local_response_normalization_across_maps(const float* in, float* denom, const int num,
			const int channels, const int height, const int width, const int size,
			const float alpha, const float beta, float* out);

	void local_response_normalization_across_maps_grad(const float* in_data, const float* out_data, const float* scale, const float* out_diff,
			const int num, const int channels, const int height, const int width, const int size, const float alpha, const float beta, float* in_diff,
			float factNew = 1.f, float factOld = 0.f);

	}
}

#endif /* CAFFE_OPS_HPP_ */
