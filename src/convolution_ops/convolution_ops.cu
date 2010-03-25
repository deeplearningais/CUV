/** 
 * @file convolution_ops.cu
 * @brief Operations used for convolution and max-pooling
 * @ingroup convolution
 * @date 2010-03-21
 */
#include <float.h>

#include "convolution_ops.hpp"

#include <convert/convert.hpp>
#include <matrix_ops/matrix_ops.hpp>
#include <conv_util.cuh>
#include <random/random.hpp>
#include <nvmatrix.cuh>
#include <conv.cuh>
#include <conv2.cuh>
#include <conv3.cuh>
#include <conv_util.cuh>
#include <convCPU.h>
#include <iostream>
using namespace std;


namespace cuv{

/**
 * Convolve N patterns (images) with F filters, resulting in N*F target images.
 *
 * @param img		contains one input pattern in each row
 * @param filters	contains one filter in each row, number of filters must
 * 			        be multiples of 16.
 * @param dst		holds the target images of the convolution. one row for each
 *			        input image. width = dstSize^2 * numFilters
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

/**
  * @copydoc convolve(dev_dense_matrix<float,row_major>&,dev_dense_matrix<float,row_major>&,dev_dense_matrix<float,row_major>&)
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

	if (!(dstSize == (imgSize - filterSize + 1)))
		std::cout << "destSize should be " << imgSize - filterSize + 1 << " but is " << dstSize;
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

// images --> blocks
template<>
void grid_to_matrix(dev_dense_matrix<float,row_major>& mat,
		  dev_dense_matrix<float,row_major>&   grid,
		  int poolSize) {
	int numImages = grid.h();
	int imgPixels = grid.w();
	int regionsPerImage = imgPixels / (poolSize * poolSize);
	int imgSize = sqrt(grid.w());

	// some preliminary checks
	cuvAssert(imgSize*imgSize == grid.w());
	cuvAssert(mat.h() == numImages*regionsPerImage);
	cuvAssert(mat.w() == poolSize*poolSize);

	// make nvMatrices with this data
	NVMatrix nv_mat(mat.ptr(), mat.h(), mat.w(), false);
	NVMatrix nv_grid(grid.ptr(), grid.h(), grid.w(), false);
	fill(mat.vec(),0);

	gridToMatrix(&nv_grid, &nv_mat, poolSize, true);

	cuvSafeCall(cudaThreadSynchronize());
}

template<>
void matrix_to_grid(dev_dense_matrix<float,row_major>& grid,
		  dev_dense_matrix<float,row_major>&   mat,
		  int poolSize) {
	int numImages = grid.h();
	int imgPixels = grid.w();
	int regionsPerImage = imgPixels / (poolSize * poolSize);
	int imgSize = sqrt(grid.w());

	// some preliminary checks
	cuvAssert(imgSize*imgSize == grid.w());
	cuvAssert(mat.h() == numImages*regionsPerImage);
	cuvAssert(mat.w() == poolSize*poolSize);

	// make nvMatrices with this data
	NVMatrix nv_mat(mat.ptr(), mat.h(), mat.w(), false);
	NVMatrix nv_grid(grid.ptr(), grid.h(), grid.w(), false);
	fill(grid.vec(),0);

	// transform and calculate maximum
	matrixToGrid(&nv_mat, &nv_grid, poolSize, true);

	cuvSafeCall(cudaThreadSynchronize());
}

template<>
void sample_multinomial(dev_dense_matrix<float,row_major>& grid){
   /*dev_dense_matrix<float,row_major> tmp(grid.h(),grid.w());*/
   /*apply_binary_functor(tmp,grid,BF_COPY);*/

   dev_dense_matrix<float,row_major> rnd(grid.h(),1);
   fill_rnd_uniform(rnd.vec());

   NVMatrix nv_grid(grid.ptr(),grid.h(),grid.w(),false);
   NVMatrix nv_rnd(rnd.ptr(),rnd.h(),rnd.w(),false);
   /*NVMatrix nv_tmp(tmp.ptr(),tmp.h(),tmp.w(),false);*/

   /*sampleMultinomial(&nv_tmp,&nv_rnd,&nv_grid); */
   sampleMultinomial(&nv_grid,&nv_rnd,&nv_grid); 
   cuvSafeCall(cudaThreadSynchronize());
}

template<>
void prob_max_pooling(dev_vector<float>& sums,dev_dense_matrix<float,row_major>& grid, int poolSize, bool sample){
	int numImages = grid.h();
	int imgPixels = grid.w();
	int regionsPerImage = imgPixels / (poolSize * poolSize);

	dev_dense_matrix<float,row_major> mat(numImages*regionsPerImage, poolSize*poolSize);
	grid_to_matrix(mat,grid,poolSize);

	// normalize rows
	reduce_to_col(sums,mat);                    // sums      = sum(mat, axis=1)
	/*apply_scalar_functor(sums,SF_ADD,1.f);      // sums     += 1*/
	apply_scalar_functor(sums,SF_INV);          // sums      = 1/sums
	matrix_times_col(mat,sums);                 // mat[:,i] *= sums

	if(sample){
		sample_multinomial(mat);
		reduce_to_col(sums,mat);                // now is 0 or 1
	}else{
		/*apply_scalar_functor(sums,SF_SMAX);             // sums      = (sums-1)/sums*/
		reduce_to_col(sums,mat);                
	}
	matrix_to_grid(grid,mat,poolSize);
}

template<>
void prob_max_pooling(dev_dense_matrix<float,row_major>& grid, int poolSize, bool sample){
	int numImages = grid.h();
	int imgPixels = grid.w();
	int regionsPerImage = imgPixels / (poolSize * poolSize);

	dev_vector<float> sums(numImages*regionsPerImage);
	prob_max_pooling(sums, grid, poolSize,sample);
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

template<>
	void convolve3(dev_dense_matrix<float,row_major>& dst,
			  dev_dense_matrix<float,row_major>&   img,
			  dev_dense_matrix<float,row_major>&   filter) {

	int numFilters = filter.h();
	int smallSize = sqrt(img.w()/numFilters);
	int filterSize = sqrt(filter.w());
	int bigSize = sqrt(dst.w());
	int numImages = img.h();

	// make NVMatrices with this data
	NVMatrix nv_dst(dst.ptr(), dst.h(), dst.w(), false);
	NVMatrix nv_img(img.ptr(), img.h(), img.w(), false);
	NVMatrix nv_filter(filter.ptr(), filter.h(), filter.w(), false);

	// execute convolution
	convolve3_bw(&nv_img, &nv_filter, &nv_dst);
	cuvSafeCall(cudaThreadSynchronize());
}

template<>
void convolve3(host_dense_matrix<float,row_major>& dst,
		  host_dense_matrix<float,row_major>&   img,
		  host_dense_matrix<float,row_major>&   filter) {
	// TODO
	printf("convolve3 NYI on host!\n");
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

__global__
void supersample_kernel(float*dst, float* src, int* indices, int len, int factor, int smallLen) {
	int tx = threadIdx.x; // ty = threadIdx.y;
	int bx = blockIdx.x, by = blockIdx.y;

	dst += by * len*len + tx * len * factor + bx * factor;
	indices += by * smallLen * smallLen + tx * smallLen + bx;
	src += by * smallLen * smallLen + tx * smallLen + bx;

	int idx = indices[0]; // coalesced???
	int row = idx % factor;
	int col = idx / factor;

	dst[row*len + col] = *src;
}

/*
 * Supersampling takes a n x (m*m) matrix img with n images of size (m x m)
 * and a factor s. Output is a n x (m*s*m*s) matrix dst with n enlarged images
 * of size (m*s x m*s)
 */
template<>
void supersample(dev_dense_matrix<float,row_major>& dst,
		dev_dense_matrix<float,row_major>& img,
		int factor,
		dev_dense_matrix<int,row_major>* indices) {
	int numImages = img.h();
	int imgPixels = img.w();
	int dstPixels = imgPixels * (factor * factor);
	int imgSize = sqrt(img.w());
	int dstSize = imgSize * factor;
	
	cuvAssert(dstSize / factor == imgSize);
	
	cuvAssert(img.w()  *factor * factor == dst.w());
	cuvAssert(img.h()== dst.h());

	NVMatrix nv_img(img.ptr(), numImages, imgPixels, false);
	NVMatrix nv_dst(dst.ptr(), numImages, dstPixels, false);

	if(indices == NULL) {
		supersample(&nv_img, &nv_dst, factor);
	} else {
		assert(imgSize < 512);
		fill(dst, 0.0f);

		dim3 grid(imgSize, img.h());
		dim3 threads(min(imgSize, 512));
		supersample_kernel<<<grid,threads>>>(dst.ptr(), img.ptr(), indices->ptr(), dstSize, factor, imgSize);
	}

	cuvSafeCall(cudaThreadSynchronize());
}

template<>
void supersample(host_dense_matrix<float,row_major>& dst,
		host_dense_matrix<float,row_major>& img,
		int factor,
		host_dense_matrix<int,row_major>* indices) {
	int numImages = img.h();
	int imgSize = sqrt(img.w());
	int dstSize = imgSize * factor;

	cuvAssert(img.w()  *factor * factor == dst.w());
	cuvAssert(img.h()== dst.h());
	cuvAssert(dstSize / factor == imgSize);

	float* image = img.ptr();
	float* target = dst.ptr();

	if(indices != NULL) {
		for(int i = 0; i < numImages; i++) {
			for(int r = 0; r < imgSize; r++)
				for(int c = 0; c < imgSize; c++) {
					int idx = (indices->vec())[r*imgSize+c + i*imgSize*imgSize];
					int row = idx % factor;
					int col = idx / factor;
					target[(r*factor+row)*dstSize + c*factor+col] = image[r*imgSize + c];
				}
			target += dst.w();
			image += img.w();
		}
	} else {
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

#define CONST_SIZE 512
__device__ __constant__ float c_filter[CONST_SIZE];

template<bool FILTER>
__global__
void super_to_max_kernel(float*dst, float* src, int* indices, int imgSize, int dstSize, int poolSize, int stepSize, int patchSize, int numPatches, int batch) {
	int tx = threadIdx.x; // ty = threadIdx.y;
	int bx = blockIdx.x;

	int patch = tx + batch * 256;

	if(patch >= numPatches * numPatches)
		return;

	int c = patch % (numPatches);
	int r = patch / (numPatches);

	dst += bx * dstSize * dstSize + c * patchSize * stepSize + r * patchSize * stepSize * dstSize;
	src += bx * imgSize * imgSize + c * patchSize + r * patchSize * imgSize;
	indices += bx * imgSize * imgSize + c * patchSize + r * patchSize * imgSize;

	for(int i=0; i<patchSize; i++) {
		for(int j=0; j<patchSize; j++) {
			if(c*patchSize+j < imgSize && r*patchSize+i < imgSize) {
				int idx = indices[0];
				int row = idx % poolSize;
				int col = idx / poolSize;
				float val = src[0];
				if(FILTER)
					val *= (float) c_filter[row*poolSize+col];
				dst[col + row*dstSize] += val;
			}

			dst += stepSize;
			src++;
			indices++;
			syncthreads();
		}
		dst += dstSize * stepSize - patchSize * stepSize;
		src += imgSize - patchSize;
		indices += imgSize - patchSize;
	}
}

template<>
void super_to_max(dev_dense_matrix<float,row_major>& dst,
		dev_dense_matrix<float,row_major>& img,
		int poolSize,
		int overlap,
		dev_dense_matrix<int,row_major>* indices,
		dev_dense_matrix<float, row_major>* filter) {
	cuvAssert(poolSize > overlap);
	int numImages = dst.h();
	cuvAssert(numImages == img.h());
	int imgSize = sqrt(img.w());
	if(imgSize * imgSize != img.w()){
			cout << std::endl<<"Error: imgSize x imgSize (" <<imgSize<<")²="<< imgSize*imgSize<<" should be img.w = "<<img.w()<<std::endl;
		}
	cuvAssert(imgSize * imgSize == img.w());
	int stepSize = poolSize - overlap;
	int dstSize = (imgSize - 1) * stepSize + poolSize;
	if(dstSize * dstSize != dst.w()){
				cout << std::endl<<"Error: dstSize x dstSize (" <<dstSize<<")²="<< dstSize*dstSize<<" should be dst.w = "<<dst.w()<<std::endl;
			}
	cuvAssert(dstSize * dstSize == dst.w());
	cuvAssert((dstSize-poolSize)/stepSize + 1 == imgSize);

	// we have to split the small image into disjoint "patches", in order to
	// avoid that data from the same patch is written to identical positions
	int patchSize = (int) ceil(((float) poolSize)/stepSize);
	int numPatches = ceil((float) imgSize / patchSize);

	if(indices == NULL) {
		printf("super_to_max() NYI without indices\n");
		return;
	}

	if(filter!=NULL) {
		cuvAssert(filter->w() == poolSize);
		cuvAssert(filter->h() == poolSize);
		cuvAssert(sizeof(float) * filter->n() <= CONST_SIZE);
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_filter, filter->ptr(), sizeof(float) * filter->n(), 0, cudaMemcpyDeviceToDevice) );
	}

	fill(dst, 0.0f);

	int numThreads = 256;
	int numBlocks = numImages;
	int numBatches = ceil((float) (numPatches * numPatches)/numThreads); // can't spread this to multiple blocks due to overlapping borders. loop instead

	for(int b = 0; b < numBatches; b++) {
		dim3 grid(numBlocks);
		dim3 threads(numThreads);
		if(filter==NULL)
			super_to_max_kernel<false><<<grid,threads>>>(dst.ptr(), img.ptr(), indices->ptr(), imgSize, dstSize, poolSize, stepSize, patchSize, numPatches, b);
		else
			super_to_max_kernel<true><<<grid,threads>>>(dst.ptr(), img.ptr(), indices->ptr(), imgSize, dstSize, poolSize, stepSize, patchSize, numPatches, b);
		cuvSafeCall(cudaThreadSynchronize());
	}
}

template<>
void super_to_max(host_dense_matrix<float,row_major>& dst,
		host_dense_matrix<float,row_major>& img,
		int poolSize,
		int overlap,
		host_dense_matrix<int,row_major>* indices,
		host_dense_matrix<float, row_major>* filter) {
	cuvAssert(poolSize > overlap);
	int numImages = dst.h();
	cuvAssert(numImages == img.h());
	int imgSize = sqrt(img.w());
	cuvAssert(imgSize * imgSize == img.w());
	int stepSize = poolSize - overlap;
	int dstSize = (imgSize - 1) * stepSize + poolSize;
	cuvAssert(dstSize * dstSize == dst.w());
	cuvAssert((dstSize-poolSize)/stepSize + 1 == imgSize);

	fill(dst, 0.0f);

	float* img_ptr = img.ptr();
	int* idx_ptr = indices->ptr();
	float* dst_ptr = dst.ptr();

	for(int i = 0; i < numImages; i++) {
		for(int r = 0; r < imgSize; r++) {
			for(int c = 0; c < imgSize; c++) {
				int idx = *idx_ptr;
				int row = idx % poolSize;
				int col = idx / poolSize;
				float val = *img_ptr;
				if(filter != NULL)
					val *= (float) (filter->ptr())[row*poolSize+col];
				dst_ptr[col + row * dstSize] += val;
				img_ptr++;
				idx_ptr++;
				dst_ptr += stepSize;
			}
			dst_ptr += overlap + (stepSize - 1) * dstSize;
		}
		dst_ptr += overlap * dstSize;
	}
}

template<>
	void copy_into(dev_dense_matrix<float,row_major>& dst,
			  dev_dense_matrix<float,row_major>&   img,
			  int padding) {
	int inputSize = sqrt(img.w());
	int outputSize = sqrt(dst.w());
	cuvAssert(inputSize * inputSize == img.w());
	cuvAssert(outputSize * outputSize == dst.w());
	cuvAssert(inputSize + 2 * padding == outputSize);
	cuvAssert(img.h() == dst.h());

	// make NVMatrices with this data
	NVMatrix nv_dst(dst.ptr(), dst.h(), dst.w(), false);
	NVMatrix nv_img(img.ptr(), img.h(), img.w(), false);

	copyInto(&nv_img, &nv_dst, padding, false);
}

template<>
	void copy_into(host_dense_matrix<float,row_major>& dst,
			  host_dense_matrix<float,row_major>&   img,
			  int padding) {
	int inputSize = sqrt(img.w());
	int outputSize = sqrt(dst.w());
	cuvAssert(inputSize * inputSize == img.w());
	cuvAssert(outputSize * outputSize == dst.w());
	cuvAssert(inputSize + 2 * padding == outputSize);
	cuvAssert(img.h() == dst.h());

	float* img_ptr = img.ptr();
	float* dst_ptr = dst.ptr();
	for(int i=0; i<img.h(); i++) {
		dst_ptr += outputSize * padding;
		for(int j=0; j<inputSize;j++) {
			dst_ptr += padding;
			for(int k=0; k<inputSize;k++) {
				*dst_ptr++ = *img_ptr++;
			}
			dst_ptr += padding;
		}
		dst_ptr += outputSize * padding;
	}
}

template<>
	void max_pooling(host_dense_matrix<float,row_major>& dst,
			host_dense_matrix<float,row_major>& img,
			unsigned int poolSize,
			unsigned int overlap,
			host_dense_matrix<int,row_major>* indices,
			host_dense_matrix<float,row_major>* filter) {
	cuvAssert(poolSize > overlap);
	//cuvAssert( dst.w() == indices.w());
	//cuvAssert( dst.h() == indices.h());
	int numImages = dst.h();
	cuvAssert(numImages == img.h());
	int imgSize = sqrt(img.w());
	cuvAssert(imgSize * imgSize == img.w());
	int stepSize = poolSize - overlap;
	int dstSize = (imgSize - poolSize)/stepSize + 1;
	cuvAssert(dstSize * dstSize == dst.w());
	cuvAssert((dstSize-1)*stepSize + poolSize == imgSize);
	if(filter!=NULL) {
		cuvAssert(filter->w() == poolSize);
		cuvAssert(filter->h() == poolSize);
	}

	float* img_ptr = img.ptr();
	float* dst_ptr = dst.ptr();

	for(int p=0; p<numImages; p++) {
		for(int r=0; r<dstSize; r++)
			for(int c=0; c<dstSize; c++) {
				int imax = 0;
				float cmax = -FLT_MAX;
				// loop through pool
				for(int i=0; i<poolSize; i++)
					for(int j=0; j<poolSize; j++) {
						int idx = c*stepSize+j + (r*stepSize+i)*imgSize;
						float val = img_ptr[idx];
						if(filter!=NULL)
							val *= (*filter)(i,j);
						if(cmax < val) {
							cmax = val;
							imax = j*poolSize+i; // transpose due to dev local_maximum() function
						}
					}
				*dst_ptr++ = cmax;
				if(indices != NULL)
					indices->set(p, r*dstSize+c, imax);
			}

		img_ptr += imgSize * imgSize;
	}
}

// naive, but flexible implementation
// better distinguish between different cases and load image into shared memory
template<bool INDEX, bool FILTER>
__global__
void max_pooling_kernel(float* dst, float* img, int* indices, int imgSize, int dstSize, int poolSize, int stepSize) {
	int tx = threadIdx.x; // ty = threadIdx.y;
	int bx = blockIdx.x, by = blockIdx.y;

	int p = tx + by * 256;
	if(p >= dstSize * dstSize)
		return;

	img += bx * imgSize * imgSize;

	float cmax = -FLT_MAX;
	int imax = 0;
	int column = p % dstSize;
	int row = p / dstSize;

	// loop through pool
	for(int i=0; i<poolSize; i++)
		for(int j=0; j<poolSize; j++) {
			int idx = column*stepSize+j + (row*stepSize+i)*imgSize;
			float val = img[idx];
			if(FILTER)
				val *= (float) c_filter[i*poolSize+j];
			if(cmax < val) {
				cmax = val;
				if(INDEX)
					imax = j*poolSize+i; // transpose due to dev local_maximum() function
			}
		}

	// write result
	dst += bx * dstSize * dstSize + p;
	//	indices
	if(INDEX) {
		indices += bx * dstSize * dstSize + p;
		*indices = imax;
	}
	*dst = cmax;
}

/* This implementation only achieves a speedup of 5-10x, and is even
 * worse if the pools do not overlap. Better use local_maximum() in this
 * case.
 */

template<>
	void max_pooling(dev_dense_matrix<float,row_major>& dst,
			dev_dense_matrix<float,row_major>& img,
			unsigned int poolSize,
			unsigned int overlap,
			dev_dense_matrix<int,row_major>* indices,
			dev_dense_matrix<float, row_major>* filter) {

	cuvAssert(poolSize > overlap);
	int numImages = dst.h();
	cuvAssert(numImages == img.h());
	int imgSize = sqrt(img.w());
	cuvAssert(imgSize * imgSize == img.w());
	int stepSize = poolSize - overlap;
	int dstSize = (imgSize - poolSize)/stepSize + 1;
	cuvAssert(dstSize * dstSize == dst.w());
	cuvAssert((dstSize-1)*stepSize + poolSize == imgSize);
	if(indices){
		cuvAssert(indices->w() == numImages);
		cuvAssert(indices->h() == imgSize*imgSize / (poolSize*poolSize));
	}

	int numThreads = 256;
	int numBlocksX = numImages;
	int numBlocksY = ceil((float) (dstSize * dstSize)/numThreads);

	if(filter!=NULL) {
		cuvAssert(filter->w() == poolSize);
		cuvAssert(filter->h() == poolSize);
		cuvAssert(sizeof(float) * filter->n() <= CONST_SIZE);
		cuvSafeCall( cudaMemcpyToSymbol(c_filter, filter->ptr(), sizeof(float) * filter->n(), 0, cudaMemcpyDeviceToDevice) );
	}
	cuvSafeCall(cudaThreadSynchronize());

	dim3 grid(numBlocksX, numBlocksY);
	dim3 threads(numThreads);
	if(indices==NULL && filter==NULL)
		max_pooling_kernel<false, false><<<grid,threads>>>(dst.ptr(), img.ptr(), NULL, imgSize, dstSize, poolSize, stepSize);
	else if(indices==NULL && filter!=NULL)
		max_pooling_kernel<false, true><<<grid,threads>>>(dst.ptr(), img.ptr(), NULL, imgSize, dstSize, poolSize, stepSize);
	else if(indices!=NULL && filter==NULL)
		max_pooling_kernel<true, false><<<grid,threads>>>(dst.ptr(), img.ptr(), indices->ptr(), imgSize, dstSize, poolSize, stepSize);
	else if(indices!=NULL && filter!=NULL)
		max_pooling_kernel<true, true><<<grid,threads>>>(dst.ptr(), img.ptr(), indices->ptr(), imgSize, dstSize, poolSize, stepSize);

	cuvSafeCall(cudaThreadSynchronize());
}

/*
 * Block size 16x16.
 */
__global__ void strip_padding_kernel(float* targets, float* images, const int imgSize, const int paddingSize, const int numImages) {
    const int imgIdx = blockIdx.y;

    //check if index is still in matrix
    if (imgIdx < numImages) {
        const int targetSize = imgSize - 2 * paddingSize;
        // move pointer by imgIdx images
        images += imgIdx * imgSize * imgSize;

        // move pointer by imgIdx images
        targets += imgIdx * targetSize * targetSize;

        // what is this pixels index in the source image
        int px = blockIdx.x * blockDim.x + threadIdx.x;

        // pixels coordinates
        int x = px % imgSize;
        int y = px / imgSize;
        if ( x >= paddingSize && x < paddingSize+targetSize &&
        	 y >= paddingSize && y < paddingSize+targetSize){
            // move source pointer to this pixels index in source umage
            images+=px;

        	// move target pointer to target position,
        	targets	+=	(y-paddingSize)*targetSize+(x-paddingSize);

        	// copy contents
        	*targets = *images;
        }
    }
}

/*
 * strip padding removes a border of padding size from each picture_row
 *
 */
template<>
	void strip_padding(dev_dense_matrix<float,row_major>& dst,
					   dev_dense_matrix<float,row_major>& img,
					   unsigned int padding) {
	int inputSize = sqrt(img.w());
	int imgWidth = inputSize;
	int outputSize = sqrt(dst.w());
	int numImages = img.h();
	cuvAssert(inputSize * inputSize == img.w());
	cuvAssert(outputSize * outputSize == dst.w());
	cuvAssert(inputSize - 2 * padding == outputSize);
	cuvAssert(img.h() == dst.h());
	int numThreads = 256;
	int numBlocksX = ceil((float) (imgWidth * imgWidth)/numThreads);
	int numBlocksY = numImages;
	dim3 grid(numBlocksX, numBlocksY);
	dim3 dimBlock(numThreads,1);
	strip_padding_kernel<<<grid,dimBlock>>>(dst.ptr(), img.ptr(), imgWidth, numImages, padding);
}

template<>
	void strip_padding(host_dense_matrix<float,row_major>& dst,
					   host_dense_matrix<float,row_major>& img,
					   unsigned int padding) {
	int inputSize = sqrt(img.w());
	int imgWidth = inputSize;
	int outputSize = sqrt(dst.w());
	int numImages = img.h();
	cuvAssert(inputSize * inputSize == img.w());
	cuvAssert(outputSize * outputSize == dst.w());
	cuvAssert(inputSize - 2 * padding == outputSize);
	cuvAssert(img.h() == dst.h());
	fill(dst, 0.0f);


	int x,y, idx, idx_padded;
	float val;
	int stripped_width = imgWidth - 2 * padding;

	for (int imgIdx = 0; imgIdx < img.h(); imgIdx++){
		for(int px = 0; px < img.w(); px++){
			x = px % inputSize;
			y = px / inputSize;
			if ( x >=padding && x < padding+stripped_width &&
				 y >=padding && y < padding+stripped_width)
			{
				idx 		=	y*inputSize+x;
				idx_padded 	=	(y-padding)*stripped_width+(x-padding);
				val = img(imgIdx, idx);
				dst.set(imgIdx, idx_padded, val);
			}
		}
	}

}

/*
 * Block size 16x16.
 */
__global__ void row_ncopy_kernel(float* targets, float* row, const int imgSize, const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //check if index is still in matrix
    if (idx < imgSize) {
    	row += idx;
    	targets += idx;
    	for(int i=0; i < n ;i++){
    	     *targets = *row;
    	     targets += imgSize;
    	}
    }
}

/*
 * copy 1st row n times in 1 one column
 */

template<>
	void row_ncopy(dev_dense_matrix<float,row_major>& dst,
				   dev_vector<float>& row,
				   unsigned int n) {
	int inputSize = row.size();
	cuvAssert(n == dst.h());
	cuvAssert(n <= 4096);
	fill(dst, 0.0f);

	int numThreads = 256;
	int numBlocksX = ceil(inputSize/numThreads);
	int numBlocksY = 1;
	dim3 grid(numBlocksX, numBlocksY);
	dim3 dimBlock(numThreads,1);
	row_ncopy_kernel<<<grid,dimBlock>>>(dst.ptr(), row.ptr(), inputSize, n);
}

}
