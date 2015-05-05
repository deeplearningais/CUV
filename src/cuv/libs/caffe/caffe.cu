//  Licensed under the BSD 2-Clause license.
//  CUV/src/cuv/libs/caffe/LICENSE

#include "caffe.hpp"

namespace cuv{

namespace caffe{


#if __CUDA_ARCH__ >= 200
const int CUDA_NUM_THREADS = 1024;
#else
const int CUDA_NUM_THREADS = 512;
#endif
// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
i < (n); \
i += blockDim.x * gridDim.x)


__global__ void empty_kernel()
{
	}

void empty_kernel_call()
{
	empty_kernel<<<1, 1>>>();

}

void create_cuda_stream(cudaStream_t &stream)
{
	cudaError_t result = cudaStreamCreate(&stream);
	if (result != cudaSuccess)
		throw("error creating cuda stream");
};


void destroy_cuda_stream(cudaStream_t &stream)
{
	cudaError_t result = cudaStreamDestroy(stream);
	if (result != cudaSuccess)
		throw("error destroying cuda stream");
};


__global__ void lrn_across_maps_fill_scale_kernel(const int nthreads, const float* in, float* scale, const int num, const int channels, const int height,
	const int width, const int size, const float alpha_over_size, const float negative_beta) {

	CUDA_KERNEL_LOOP(index, nthreads) {
	// find out the local offset
		int w = index % width;
		int h = (index / width) % height;
		int n = index / width / height;
		int offset = (n * channels * height + h) * width + w;
		int step = height * width;
		in += offset;
		scale += offset;
		int head = 0;
		int pre_pad = (size - 1) / 2;
		int post_pad = size - pre_pad - 1;
		float accum_scale = 0;

		// fill the scale at [n, :, h, w]
		// accumulate values
		while (head < post_pad) {
			accum_scale += in[head * step] * in[head * step];
			++head;
		}
		// until we reach size, nothing needs to be subtracted
		while (head < size) {
			accum_scale += in[head * step] * in[head * step];
			scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;

			++head;
		}
		// both add and subtract
		while (head < channels) {
			accum_scale += in[head * step] * in[head * step];
			accum_scale -= in[(head - size) * step] * in[(head - size) * step];
			scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
			++head;
		}
		// subtract only
		while (head < channels + post_pad) {
			accum_scale -= in[(head - size) * step] * in[(head - size) * step];
			scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
			++head;
		}
	}
}

__global__ void local_response_normalization_across_maps_grad_kernel(const int nthreads, const float* in_data, const float* out_data, const float* scale, const float* out_diff,
const int num, const int channels, const int height, const int width, const int size, const float negative_beta, const float cache_ratio, float* in_diff,
float factNew, float factOld) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		// find out the local offset
		int w = index % width;
		int h = (index / width) % height;
		int n = index / width / height;
		int offset = (n * channels * height + h) * width + w;
		int step = height * width;
		in_data += offset;
		out_data += offset;
		scale += offset;
		out_diff += offset;
		in_diff += offset;
		int head = 0;
		int pre_pad = size - (size + 1) / 2;
		int post_pad = size - pre_pad - 1;
		float accum_ratio = 0;
		// accumulate values
		while (head < post_pad) {
			accum_ratio += out_diff[head * step] * out_data[head * step] /
			scale[head * step];
			++head;
		}
		// until we reach size, nothing needs to be subtracted
		while (head < size) {
			accum_ratio += out_diff[head * step] * out_data[head * step] /
			scale[head * step];
			in_diff[(head - post_pad) * step] = factOld * in_diff[(head - post_pad) * step] + factNew * (out_diff[(head - post_pad) * step]
			* pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
			in_data[(head - post_pad) * step] * accum_ratio);
			++head;
		}
		// both add and subtract
		while (head < channels) {
			accum_ratio += out_diff[head * step] * out_data[head * step] /
			scale[head * step];
			accum_ratio -= out_diff[(head - size) * step] *
			out_data[(head - size) * step] / scale[(head - size) * step];
			in_diff[(head - post_pad) * step] = factOld * in_diff[(head - post_pad) * step] + factNew * (out_diff[(head - post_pad) * step]
			* pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
			in_data[(head - post_pad) * step] * accum_ratio);
			++head;
		}
		// subtract only
		while (head < channels + post_pad) {
			accum_ratio -= out_diff[(head - size) * step] *
			out_data[(head - size) * step] / scale[(head - size) * step];
			in_diff[(head - post_pad) * step] = factOld * in_diff[(head - post_pad) * step] + factNew * (out_diff[(head - post_pad) * step]
			* pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
			in_data[(head - post_pad) * step] * accum_ratio);
			++head;
		}
	}
}

__global__ void lrn_across_maps_compute_output_kernel(const int nthreads, const float* in,
		const float* scale, const float negative_beta, float* out) {
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		out[index] = in[index] * pow(scale[index], negative_beta);
	}
}

void local_response_normalization_across_maps(const float* in, float* scale,
		const int num, const int channels, const int height,
		const int width, const int size, const float alpha, const float beta,
		float* out)
{
	assert(size<=channels);
        // if size can be larger, then this fix: https://github.com/BVLC/caffe/pull/1922 should be implemented.

	int n_threads = num * height * width;
	const float alpha_over_size = alpha / size;

	int nblocks = GET_BLOCKS(n_threads);
    	lrn_across_maps_fill_scale_kernel<<<nblocks, CUDA_NUM_THREADS>>>(n_threads, in, scale, num, channels, height, width, size,
	    alpha_over_size, -beta);
    	n_threads = num * channels * height * width;
    	lrn_across_maps_compute_output_kernel<<<nblocks, CUDA_NUM_THREADS>>>(n_threads, in, scale, -beta, out);

}

void local_response_normalization_across_maps_grad(const float* in_data, const float* out_data, const float* scale, const float* out_diff,
		const int num, const int channels, const int height, const int width, const int size, const float alpha, const float beta,
		float* in_diff, float factNew, float factOld)
{
	 int n_threads = num * height * width;


     int nblocks = GET_BLOCKS(n_threads);
	 local_response_normalization_across_maps_grad_kernel<<<nblocks, CUDA_NUM_THREADS>>>(
	n_threads, in_data, out_data,
	scale, out_diff, num, channels, height, width,
	size, -beta, (float)(2. * alpha * beta/ size),
	in_diff, factNew, factOld);

}

}
}
