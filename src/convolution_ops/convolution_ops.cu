#include "convolution_ops.hpp"

#include <convert/convert.hpp>
#include <nvmatrix.cuh>
#include <conv.cuh>
#include <conv_util.cuh>

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

	cudaThreadSynchronize();
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
	// NYI
	printf("warning! resorting on dev not yet implemented!\n");

	cudaThreadSynchronize();
}

template<>
void reorder(host_dense_matrix<float,row_major>& M,
		  int blockLength) {
	int patternCount = M.h();
	int imgSize = M.w()/blockLength;

	float* temp = (float*) malloc(sizeof(float) * M.n());
	float* img = M.ptr();

	for(int p = 0; p < patternCount; p++)
		for(int m = 0; m < blockLength; m++) {
			memcpy(	&temp[p*imgSize+m*imgSize*patternCount],
					&img[p*M.w()+m*imgSize], sizeof(float)*imgSize);
		}

	memcpy(M.ptr(), temp, sizeof(float) * M.n());
	M.resize(patternCount*blockLength, imgSize);
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

	NVMatrix nv_img(img.ptr(), numImages, imgPixels, false);
	NVMatrix nv_dst(dst.ptr(), numImages, dstPixels, false);

	supersample(&nv_img, &nv_dst, factor);

	cudaThreadSynchronize();
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
