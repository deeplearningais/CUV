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

	for(int i=0; i<numImages; i++)
		for(int r=0; r<imgSize; r++)
			for(int c=0; c<imgSize; c++)
				for(int f=0; f<numFilters; f++) {
					float sum = 0.0f;
					for(int y=0; y<filterSize; y++)
						for(int x=0; x<filterSize; x++)
							sum += img(i, (r+y)*imgSize + (c+x) ) * filter(f, y * filterSize + x);
					dst.set(i, f*dstPixels + r*dstSize + c, sum);
				}

	/* obsolete code of host convolution utilizing cuda devices

	// some preliminary checks to ensure compatibility
	int numFilters = filter.h();
	int filterSize = sqrt(filter.w());
	int imgSize = sqrt(img.w());
	int dstSize = sqrt(dst.w()/numFilters);
	cuvAssert(numFilters%16 == 0);
	cuvAssert(filterSize*filterSize == filter.w());
	cuvAssert(imgSize*imgSize == img.w());
	cuvAssert(dstSize == imgSize - filterSize + 1);

	// make copies of the host matrices on the device
	dev_dense_matrix<float,row_major> d_img(img.h(),img.w());
	convert(d_img, img);
	dev_dense_matrix<float,row_major> d_dst(dst.h(),dst.w());
	fill(d_dst.vec(), 0.0f);
	dev_dense_matrix<float,row_major> d_filter(filter.h(),filter.w());
	convert(d_filter, filter);

	// make NVMatrices with this data
	NVMatrix nv_dst(d_dst.ptr(), d_dst.h(), d_dst.w(), false);
	NVMatrix nv_img(d_img.ptr(), d_img.h(), d_img.w(), false);
	NVMatrix nv_filter(d_filter.ptr(), d_filter.h(), d_filter.w(), false);

	// execute convolution
	convolve_bw(&nv_img, &nv_filter, &nv_dst);

	// convert back
	convert(dst, d_dst);
	*/
}

}
