#include "convolution_ops.hpp"

#include <convert/convert.hpp>
#include <nvmatrix.cuh>
#include <conv.cuh>
#include <conv2.cuh>
#include <conv_util.cuh>
#include <convCPU.h>

namespace cuv{

/* Convolve N patterns (images) with F filters, resulting in N*F target images
 *
 * img		contains one input pattern in each row
 * filters	contains one filter in each row, number of filters must
 * 			be multiples of 16.
 * dst		holds the target images of the convolution. one row for each
 *			input image. width = dstSize^2 * numFilters
 */
template<>
	void convolve(dev_dense_matrix<float,row_major>& dst,
			  dev_dense_matrix<float,row_major>&   img,
			  dev_dense_matrix<float,row_major>&   filter){

	// some preliminary checks to ensure compatibility
	int numFilters = filter.h();
	int filterSize = sqrt(filter.w());
	int imgSize = sqrt(img.w());
	int dstSize = sqrt(dst.w()/numFilters);
	cuvAssert(numFilters%16 == 0);
	cuvAssert(filterSize*filterSize == filter.w());
	cuvAssert(imgSize*imgSize == img.w());
	cuvAssert(dstSize == imgSize - filterSize + 1);

	// make NVMatrices with this data
	NVMatrix nv_dst(dst.ptr(), dst.h(), dst.w(), false);
	NVMatrix nv_img(img.ptr(), img.h(), img.w(), false);
	NVMatrix nv_filter(filter.ptr(), filter.h(), filter.w(), false);

	// execute convolution
	convolve_bw(&nv_img, &nv_filter, &nv_dst);
	cuvSafeCall(cudaThreadSynchronize());
	}

template<>
void convolve(host_dense_matrix<float,row_major>& dst,
		  host_dense_matrix<float,row_major>&   img,
		  host_dense_matrix<float,row_major>&   filter) {

	int numImages = img.h();
	int numFilters = filter.h();

	int filterSize = sqrt(filter.w());
	int imgSize = sqrt(img.w());
	int dstSize = sqrt(dst.w()/numFilters);

	int dstPixels = dstSize * dstSize;

	float* images = img.ptr();
	float* targets = dst.ptr();

	for(int i=0; i<numImages; i++) {
		float* filters = filter.ptr();
		for(int f=0; f<numFilters; f++) {
			for(int r=0; r<dstSize; r++)
				for(int c=0; c<dstSize; c++) {
					float sum = 0.0f;
					for(int y=0; y<filterSize; y++) {
						float subsum = 0.0f;
						for(int x=0; x<filterSize; x++)
							subsum += images[(r+y)*imgSize + (c+x)] * filters[y * filterSize + x];
						sum += subsum;
					}
					targets[f*dstPixels + r*dstSize + c] += sum;
				}
			filters += filter.w();
		}
		targets += dst.w();
		images += img.w();
	}
}

/* Convolve N patterns (images), each with a different set of F filters,
 * resulting in N*F target images
 *
 * img		contains one input pattern in each row
 * filters	contains F filters in each row, number of filters must
 * 			be multiples of 16.
 * dst		holds the target images of the convolution. one row for each
 *			input image. width = dstSize^2 * numFilters
 *
 * This routine can be used to compute the weight gradients: img contains the
 * activations from the lower layers filters are the error maps from the upper
 * layer. dst will then contain weight gradients for each pattern per row (sum
 * each column up).
 */
template<>
	void convolve2(dev_dense_matrix<float,row_major>& dst,
			  dev_dense_matrix<float,row_major>&   img,
			  dev_dense_matrix<float,row_major>&   filter,
			  int numFilters) {
	int imgSize = sqrt(img.w());
	int numImages = img.h();
	int filterSize = sqrt(filter.w()/numFilters);
	int dstSize = sqrt(dst.w()/numFilters);

	// some preliminary checks to ensure compatibility
	cuvAssert(numFilters%16 == 0);
	cuvAssert(filterSize*filterSize*numFilters == filter.w());
	cuvAssert(imgSize*imgSize == img.w());
	cuvAssert(dstSize == imgSize - filterSize + 1);

	// make NVMatrices with this data
	NVMatrix nv_dst(dst.ptr(), dst.h(), dst.w(), false);
	NVMatrix nv_img(img.ptr(), img.h(), img.w(), false);
	NVMatrix nv_filter(filter.ptr(), filter.h(), filter.w(), false);

	// execute convolution
    convolve2_bw(&nv_img, &nv_filter, &nv_dst, filterSize);
	cuvSafeCall(cudaThreadSynchronize());
}

template<>
void convolve2(host_dense_matrix<float,row_major>& dst,
		  host_dense_matrix<float,row_major>&   img,
		  host_dense_matrix<float,row_major>&   filter,
		  int numFilters) {
	int imgSize = sqrt(img.w());
	int numImages = img.h();
	int filterSize = sqrt(filter.w()/numFilters);
	int dstSize = sqrt(dst.w()/numFilters);

	conv2CPU(img.ptr(), filter.ptr(), dst.ptr(), imgSize, filterSize, numImages, numFilters);
}


/* Convolve N patterns, each consisting of F images/maps with F filters and add
 * them up. Resulting in N target images
 *
 * img		contains F input pattern in each row
 * filters	contains one filter in each row, number of filters must
 * 			be multiples of 16.
 * dst		holds the target image of the convolution. one row for each
 *			input image. width = dstSize^2
 */
// NYI conv3


template<>
void localMaximum(dev_dense_matrix<float,row_major>& dst,
		  dev_dense_matrix<float,row_major>&   img,
		  int poolSize) {
	int numImages = img.h();
	int imgPixels = img.w();
	int regionsPerImage = imgPixels / (poolSize * poolSize);
	int imgSize = sqrt(img.w());
	int dstSize = sqrt(dst.w());

	// some preliminary checks
	cuvAssert(imgSize*imgSize == img.w());
	cuvAssert(dstSize*dstSize == dst.w());
	cuvAssert(img.h() == dst.h());

	// make nvMatrices with this data
	NVMatrix nv_img(img.ptr(), numImages, imgPixels, false);
	NVMatrix nv_trans(numImages * regionsPerImage, poolSize * poolSize, false);
	NVMatrix nv_dst(dst.ptr(), numImages * regionsPerImage, 1, false);
	nv_trans.apply(NVMatrix::ZERO);
	nv_dst.apply(NVMatrix::ZERO);

	// transform and calculate maximum
	gridToMatrix(&nv_img, &nv_trans, poolSize, true);
	nv_trans.max(1, nv_dst);

	cuvSafeCall(cudaThreadSynchronize());
}

template<>
void localMaximum(host_dense_matrix<float,row_major>& dst,
		  host_dense_matrix<float,row_major>&   img,
		  int poolSize) {
	int numImages = img.h();
	int imgPixels = img.w();
	int imgSize = sqrt(img.w());
	int dstSize = sqrt(dst.w());

	for(int i=0; i < img.h(); i++)
		for(int r=0; r<dstSize; r++)
			for(int c=0; c<dstSize; c++) {
				float maxi = -1000.f;
				for(int y=0; y<poolSize; y++)
					for(int x=0; x<poolSize; x++) {
						float val = img(i, (r*poolSize+y)*imgSize + c*poolSize+x);
						if(maxi < val)
							maxi = val;
					}
				dst.set(i, r*dstSize+c, maxi);
			}
}

__global__
void reorder_kernel(float*dst, float* src, int len) {
	int tx = threadIdx.x, ix = blockIdx.x;
	int pt = blockIdx.y;

	if(tx >= len)
		return;
	while(tx < len) {
		dst[tx + pt * len + gridDim.y * len * ix] = src[tx + ix * len + pt * len * gridDim.x];
		tx += blockDim.x;
	}
}

/* sort the images in a matrix in a different order
 * input:  A1 B1 C1 D1
 *         A2 B2 C2 D2
 *         A3 B3 C3 D3
 * , where A1 is an image with blockLength pixels
 * output: A1
 *         A2
 *         A3
 *         B1
 *         B2
 *         ..
 */
template<>
void reorder(dev_dense_matrix<float,row_major>& M,
		  int blockLength) {
	int patternCount = M.h();
	int imgCount = M.w()/blockLength;

	float* temp;
	cuvSafeCall(cudaMalloc( (void**) &temp, sizeof(float) * M.n() ));
	float* tmp_ptr = temp;
	float* img_ptr = M.ptr();

	dim3 grid(imgCount, patternCount);
	dim3 threads(min(blockLength, 512));
	reorder_kernel<<<grid,threads>>>(temp, M.ptr(), blockLength);

	cuvSafeCall(cudaThreadSynchronize());

	cuvSafeCall(cudaMemcpy(M.ptr(), temp, sizeof(float) * M.n(),cudaMemcpyDeviceToDevice));
	M.resize(patternCount*imgCount, blockLength);
	cuvSafeCall(cudaFree(temp));

	cuvSafeCall(cudaThreadSynchronize());
}

template<>
void reorder(host_dense_matrix<float,row_major>& M,
		  int blockLength) {
	int patternCount = M.h();
	int imgCount = M.w()/blockLength;

	float* temp = (float*) malloc(sizeof(float) * M.n());
	float* tmp_ptr = temp;
	float* img_ptr = M.ptr();

	for(int p = 0; p < patternCount; p++) {
		for(int m = 0; m < imgCount; m++) {
			memcpy(	&tmp_ptr[blockLength * patternCount * m],
					img_ptr, sizeof(float)*blockLength);
			img_ptr += blockLength;
		}
		tmp_ptr += blockLength;
	}

	memcpy(M.ptr(), temp, sizeof(float) * M.n());
	M.resize(patternCount*imgCount, blockLength);
	free(temp);
}

/*
 * Supersampling takes a n x (m*m) matrix img with n images of size (m x m)
 * and a factor s. Output is a n x (m*s*m*s) matrix dst with n enlarged images
 * of size (m*s x m*s)
 */
template<>
void supersample(dev_dense_matrix<float,row_major>& dst,
		dev_dense_matrix<float,row_major>& img,
		int factor) {
	int numImages = img.h();
	int imgPixels = img.w();
	int dstPixels = imgPixels * (factor * factor);
	int imgSize = sqrt(img.w());
	int dstSize = imgSize * factor;

	cuvAssert(dstSize / factor == imgSize);

	NVMatrix nv_img(img.ptr(), numImages, imgPixels, false);
	NVMatrix nv_dst(dst.ptr(), numImages, dstPixels, false);

	supersample(&nv_img, &nv_dst, factor);

	cuvSafeCall(cudaThreadSynchronize());
}

template<>
void supersample(host_dense_matrix<float,row_major>& dst,
		host_dense_matrix<float,row_major>& img,
		int factor) {
	int numImages = img.h();
	int imgSize = sqrt(img.w());
	int dstSize = imgSize * factor;

	cuvAssert(dstSize / factor == imgSize);

	float* image = img.ptr();
	float* target = dst.ptr();

	for(int i = 0; i < numImages; i++) {
		for(int r = 0; r < dstSize; r++)
			for(int c = 0; c < dstSize; c++) {
				target[0] = image[(r/factor)*imgSize+c/factor];
				target++;
			}
		image += img.w();
	}
}

}
