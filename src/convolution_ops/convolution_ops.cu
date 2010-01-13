#include "convolution_ops.hpp"

#include <convert.hpp>
#include <nvmatrix.cuh>
#include <conv.cuh>

namespace cuv{

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

/*
 * img		contains one input pattern in each row
 * filters	contains one filter in each row, number of filters must
 * 			be multiples of 16.
 * dst		holds the target images of the convolution. one row for each
 *			input image. width = dstSize^2 * numFilters
 */
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
	int imgPixels = imgSize * imgSize;
	int filterPixels = filterSize * filterSize;

// Alex' host convolution is ~25% faster due to better memory access patterns
//	convCPU(img.ptr(), filter.ptr(), dst.ptr(), imgSize, filterSize, numImages, numFilters);

	for(int i=0; i<numImages; i++)
		for(int f=0; f<numFilters; f++)
			for(int r=0; r<dstSize; r++)
				for(int c=0; c<dstSize; c++) {
					float sum = 0.0f;
					for(int y=0; y<filterSize; y++) {
						float subsum = 0.0f;
						for(int x=0; x<filterSize; x++)
							subsum += img(i, (r+y)*imgSize + (c+x) ) *  filter(f, y * filterSize + x);
						sum += subsum;
					}
					sum += dst(i, f*dstPixels + r*dstSize + c);
					dst.set(i, f*dstPixels + r*dstSize + c, sum);
				}
}

}
