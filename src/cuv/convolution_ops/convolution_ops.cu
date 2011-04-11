//*LB*
// Copyright (c) 2010, University of Bonn, Institute for Computer Science VI
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of the University of Bonn 
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*LE*





/** 
 * @file convolution_ops.cu
 * @brief Operations used for convolution and max-pooling
 * @ingroup convolution
 * @date 2010-03-21
 */
#include <float.h>

#include <cuv/convolution_ops/convolution_ops.hpp>

#include <cuv/convert/convert.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/random/random.hpp>
#include <3rd_party/CudaConv/conv_util.cuh>
#include <3rd_party/CudaConv/nvmatrix.cuh>
#include <3rd_party/CudaConv/conv.cuh>
#include <3rd_party/CudaConv/conv2.cuh>
#include <3rd_party/CudaConv/conv3.cuh>
#include <3rd_party/CudaConv/conv_util.cuh>
#include <3rd_party/CudaConv/convCPU.h>
#include <iostream>
using namespace std;
#ifdef __CDT_PARSER__
#define __shared__
#define __global__
#endif

namespace cuv{

template<>
	void convolve(tensor<float,dev_memory_space,row_major>& dst,
			  tensor<float,dev_memory_space,row_major>&   img,
			  tensor<float,dev_memory_space,row_major>&   filter,
			  int numGroups){

	// some preliminary checks to ensure compatibility
	int numFilters = filter.shape()[0] / numGroups;
	cuvAssert(filter.shape()[0] == numFilters*numGroups);
	int numImages = img.shape()[0] / numGroups;
	cuvAssert(img.shape()[0] == numImages*numGroups);
	int filterSize = sqrt(filter.shape()[1]);
	int imgSize = sqrt(img.shape()[1]);
	int dstSize = sqrt(dst.shape()[1]/numImages);
//	printf("imgSize = %i, dstSize = %i, filterSize = %i\n", imgSize, dstSize, filterSize);
//	printf("(%ix%i) x (%ix%i) = (%ix%i)\n", img.shape()[0], img.shape()[1], filter.shape()[0], filter.shape()[1], dst.shape()[0], dst.shape()[1]);
	cuvAssert(numFilters%2 == 0);
	cuvAssert(filterSize*filterSize == filter.shape()[1]);
	cuvAssert(imgSize*imgSize == img.shape()[1]);
	cuvAssert(dstSize == imgSize - filterSize + 1);

	// make NVMatrices with this data
	NVMatrix nv_dst(dst.ptr(), dst.shape()[0], dst.shape()[1], false);
	NVMatrix nv_img(img.ptr(), img.shape()[0], img.shape()[1], false);
	NVMatrix nv_filter(filter.ptr(), filter.shape()[0], filter.shape()[1], false);

	// execute convolution
	convolve(&nv_img, &nv_filter, &nv_dst, numGroups, false);
	cuvSafeCall(cudaThreadSynchronize());
	}

template<>
void convolve(tensor<float,host_memory_space,row_major>& dst,
		  tensor<float,host_memory_space,row_major>&   img,
		  tensor<float,host_memory_space,row_major>&   filter,
		  int numGroups) {

	int numImages = img.shape()[0] / numGroups;
	int numFilters = filter.shape()[0] / numGroups;

	int filterSize = sqrt(filter.shape()[1]);
	int imgSize = sqrt(img.shape()[1]);
	int dstSize = sqrt(dst.shape()[1]/numImages);

	int dstPixels = dstSize * dstSize;

	float* targets = dst.ptr();

	for(int g=0; g<numGroups; g++) {
		float* filters = filter.ptr() + g*numGroups*filterSize;
		for(int f=0; f<numFilters; f++) {
			float* images = img.ptr();
			for(int i=0; i<numImages; i++) {
				for(int r=0; r<dstSize; r++)
					for(int c=0; c<dstSize; c++) {
						float sum = 0.0f;
						for(int y=0; y<filterSize; y++) {
							float subsum = 0.0f;
							for(int x=0; x<filterSize; x++)
								subsum += images[(r+y)*imgSize + (c+x)] * filters[y * filterSize + x];
							sum += subsum;
						}
						targets[i*dstPixels + r*dstSize + c] += sum;
					}
				images += img.shape()[1];
			}
			targets += dst.shape()[1];
			filters += filter.shape()[1];
		}
	}
}


template<>
	void convolve2(tensor<float,dev_memory_space,row_major>& dst,
			  tensor<float,dev_memory_space,row_major>&   img,
			  tensor<float,dev_memory_space,row_major>&   filter,
			  int numFilters,
			  int numGroups) {
	int imgSize = sqrt(img.shape()[1]);
	int numImages = img.shape()[0] / numGroups;
	int filterSize = sqrt(filter.shape()[1]/numImages);
	int dstSize = sqrt(dst.shape()[1]/numFilters);

	// some preliminary checks to ensure compatibility
	cuvAssert(filter.shape()[0] == numFilters*numGroups);
	cuvAssert(numFilters%2 == 0);
	cuvAssert(numImages*filterSize*filterSize == filter.shape()[1]);
	cuvAssert(imgSize*imgSize == img.shape()[1]);
	if (!(dstSize == (imgSize - filterSize + 1)))
		std::cout << "destSize should be " << imgSize - filterSize + 1 << " but is " << dstSize;
	cuvAssert(dstSize == imgSize - filterSize + 1);

	// make NVMatrices with this data
	NVMatrix nv_dst(dst.ptr(), dst.shape()[0], dst.shape()[1], false);
	NVMatrix nv_img(img.ptr(), img.shape()[0], img.shape()[1], false);
	NVMatrix nv_filter(filter.ptr(), filter.shape()[0], filter.shape()[1], false);

	// execute convolution
    convolve2(&nv_img, &nv_filter, &nv_dst, filterSize, numGroups, false);
	cuvSafeCall(cudaThreadSynchronize());
}

template<>
void convolve2(tensor<float,host_memory_space,row_major>& dst,
		  tensor<float,host_memory_space,row_major>&   img,
		  tensor<float,host_memory_space,row_major>&   filter,
		  int numFilters,
		  int numGroups) {
	int imgSize = sqrt(img.shape()[1]);
	int numImages = img.shape()[0];
	int filterSize = sqrt(filter.shape()[1]/numImages);
	int dstSize = sqrt(dst.shape()[1]/numFilters);

	conv2CPU(img.ptr(), filter.ptr(), dst.ptr(), imgSize, filterSize, numImages, numFilters, numGroups);
}

// images --> blocks
template<>
void grid_to_matrix(tensor<float,dev_memory_space,row_major>& mat,
		  tensor<float,dev_memory_space,row_major>&   grid,
		  int poolSize) {
	int numImages = grid.shape()[0];
	int imgPixels = grid.shape()[1];
	int regionsPerImage = imgPixels / (poolSize * poolSize);
	int imgSize = sqrt(grid.shape()[1]);

	// some preliminary checks
	cuvAssert(imgSize*imgSize == grid.shape()[1]);
	cuvAssert(mat.shape()[0] == numImages*regionsPerImage);
	cuvAssert(mat.shape()[1] == poolSize*poolSize);

	// make nvMatrices with this data
	NVMatrix nv_mat(mat.ptr(), mat.shape()[0], mat.shape()[1], false);
	NVMatrix nv_grid(grid.ptr(), grid.shape()[0], grid.shape()[1], false);
	fill(mat,0);

	gridToMatrix(&nv_grid, &nv_mat, poolSize, true);

	cuvSafeCall(cudaThreadSynchronize());
}

template<>
void matrix_to_grid(tensor<float,dev_memory_space,row_major>& grid,
		  tensor<float,dev_memory_space,row_major>&   mat,
		  int poolSize) {
	int numImages = grid.shape()[0];
	int imgPixels = grid.shape()[1];
	int regionsPerImage = imgPixels / (poolSize * poolSize);
	int imgSize = sqrt(grid.shape()[1]);

	// some preliminary checks
	cuvAssert(imgSize*imgSize == grid.shape()[1]);
	cuvAssert(mat.shape()[0] == numImages*regionsPerImage);
	cuvAssert(mat.shape()[1] == poolSize*poolSize);

	// make nvMatrices with this data
	NVMatrix nv_mat(mat.ptr(), mat.shape()[0], mat.shape()[1], false);
	NVMatrix nv_grid(grid.ptr(), grid.shape()[0], grid.shape()[1], false);
	fill(grid,0);

	// transform and calculate maximum
	matrixToGrid(&nv_mat, &nv_grid, poolSize, true);

	cuvSafeCall(cudaThreadSynchronize());
}

template<>
void sample_multinomial(tensor<float,dev_memory_space,row_major>& grid){
   /*dense_matrix<float,dev_memory_space,row_major> tmp(grid.shape()[0],grid.shape()[1]);*/
   /*apply_binary_functor(tmp,grid,BF_COPY);*/

   tensor<float,dev_memory_space,row_major> rnd(grid.shape()[0]);
   fill_rnd_uniform(rnd);

   NVMatrix nv_grid(grid.ptr(),grid.shape()[0],grid.shape()[1],false);
   NVMatrix nv_rnd(rnd.ptr(),rnd.shape()[0],rnd.shape()[1],false);
   /*NVMatrix nv_tmp(tmp.ptr(),tmp.shape()[0],tmp.shape()[1],false);*/

   /*sampleMultinomial(&nv_tmp,&nv_rnd,&nv_grid); */
   sampleMultinomial(&nv_grid,&nv_rnd,&nv_grid); 
   cuvSafeCall(cudaThreadSynchronize());
}

template<>
void prob_max_pooling(tensor<float,dev_memory_space>& sums,tensor<float,dev_memory_space,row_major>& grid, int poolSize, bool sample){
	int numImages = grid.shape()[0];
	int imgPixels = grid.shape()[1];
	int regionsPerImage = imgPixels / (poolSize * poolSize);

	tensor<float,dev_memory_space,row_major> mat(extents[numImages*regionsPerImage][poolSize*poolSize]);
	grid_to_matrix(mat,grid,poolSize);

	// normalize rows
	reduce_to_col(sums,mat);                    // sums      = sum(mat, axis=1)
	apply_scalar_functor(sums,SF_ADD,1.f);      // sums     += 1
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
void prob_max_pooling(tensor<float,dev_memory_space,row_major>& grid, int poolSize, bool sample){
	int numImages = grid.shape()[0];
	int imgPixels = grid.shape()[1];
	int regionsPerImage = imgPixels / (poolSize * poolSize);

	tensor<float,dev_memory_space> sums(numImages*regionsPerImage);
	prob_max_pooling(sums, grid, poolSize,sample);
}


template<>
	void convolve3(tensor<float,dev_memory_space,row_major>& dst,
			  tensor<float,dev_memory_space,row_major>&   img,
			  tensor<float,dev_memory_space,row_major>&   filter,
			  int numGroups) {

	int numFilters = filter.shape()[0];
	int smallSize = sqrt(img.shape()[1]/numFilters);
	int filterSize = sqrt(filter.shape()[1]);
	int bigSize = sqrt(dst.shape()[1]);
	int numImages = img.shape()[0];

	// make NVMatrices with this data
	NVMatrix nv_dst(dst.ptr(), dst.shape()[0], dst.shape()[1], false);
	NVMatrix nv_img(img.ptr(), img.shape()[0], img.shape()[1], false);
	NVMatrix nv_filter(filter.ptr(), filter.shape()[0], filter.shape()[1], false);

	// execute convolution
	convolve3(&nv_img, &nv_filter, &nv_dst, numGroups, false);
	cuvSafeCall(cudaThreadSynchronize());
}

template<>
void convolve3(tensor<float,host_memory_space,row_major>& dst,
		  tensor<float,host_memory_space,row_major>&   img,
		  tensor<float,host_memory_space,row_major>&   filter,
		  int numGroups) {
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

template<class V>
void reorder_impl(tensor<V,dev_memory_space,row_major>& dst,
				  tensor<V,dev_memory_space,row_major>& src,
		  		  int blockLength) {
	int patternCount = src.shape()[0];
	int imgCount = src.shape()[1]/blockLength;

	dim3 grid(imgCount, patternCount);
	dim3 threads(min(blockLength, 512));
	reorder_kernel<<<grid,threads>>>(dst.ptr(), src.ptr(), blockLength);

	cuvSafeCall(cudaThreadSynchronize());

	dst.reshape(extents[patternCount*imgCount][blockLength]);

	cuvSafeCall(cudaThreadSynchronize());
}

template<class V>
void reorder_impl(tensor<V,host_memory_space,row_major>& dst,
		tensor<V,host_memory_space,row_major>& src,
		int blockLength) {
	int patternCount = src.shape()[0];
	int imgCount = src.shape()[1]/blockLength;

	float* dst_ptr = dst.ptr();
	float* src_ptr = src.ptr();

	for(int p = 0; p < patternCount; p++) {
		for(int m = 0; m < imgCount; m++) {
			memcpy(	&dst_ptr[blockLength * patternCount * m],
					src_ptr, sizeof(float)*blockLength);
			src_ptr += blockLength;
		}
		dst_ptr += blockLength;
	}

	dst.reshape(extents[patternCount*imgCount][blockLength]);
}

template<class V, class M, class T>
void reorder(tensor<V,M,T>& m,
		int blockLength) {
	// create temporary destination matrix
	tensor<V,M,T> tmp(m.shape());

	// perform reorder
	reorder_impl(tmp, m, blockLength);

	// change pointer to temp matrix / copy
        m = tmp;
}

template<class V, class M, class T>
void reorder(tensor<V,M,T>& dst,
		tensor<V,M,T>& src,
		int blockLength) {
	reorder_impl(dst, src, blockLength);
}

#define REORDER_INSTANTIATE(V) \
	template void reorder( tensor<V,host_memory_space,row_major>&, int); \
	template void reorder( tensor<V,host_memory_space,row_major>&, tensor<V,host_memory_space,row_major>&, int); \
	template void reorder( tensor<V,dev_memory_space,row_major>&, int); \
	template void reorder( tensor<V,dev_memory_space,row_major>&, tensor<V,dev_memory_space,row_major>&, int);

REORDER_INSTANTIATE(float);

template<>
	void subsample(tensor<float,dev_memory_space,row_major>& dst,
			  tensor<float,dev_memory_space,row_major>&   img,
			  int factor,
			  bool avoidBankConflicts) {
	// make NVMatrices with this data
	NVMatrix nv_dst(dst.ptr(), dst.shape()[0], dst.shape()[1], false);
	NVMatrix nv_img(img.ptr(), img.shape()[0], img.shape()[1], false);

	if (dst.shape()[1]*dst.shape()[0] != img.shape()[1]* img.shape()[0] / (factor*factor)){
		std::cout << dst.shape()[1] << "*" << dst.shape()[0] << "==" << img.shape()[1] << "*" << img.shape()[0] << "/" << factor << "*" << factor << "==" << dst.shape()[1]*dst.shape()[0] << "!=" << img.shape()[1]* img.shape()[0] / (factor*factor);
	}

	cuvAssert(dst.shape()[1]*dst.shape()[0] == img.shape()[1]* img.shape()[0] / (factor*factor));
	// execute convolution
    subsample(&nv_img, &nv_dst, factor, avoidBankConflicts);
	cuvSafeCall(cudaThreadSynchronize());
}

template<>
	void subsample(tensor<float,host_memory_space,row_major>& dst,
			  tensor<float,host_memory_space,row_major>&   img,
			  int factor,
			  bool avoidBankConflicts) {
	int imgSize = sqrt(img.shape()[1]);
	int numImg = img.shape()[0];

	// execute convolution
    subsampleCPU(img.ptr(), dst.ptr(), imgSize, factor, numImg);
	cuvSafeCall(cudaThreadSynchronize());
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

template<>
void supersample(tensor<float,dev_memory_space,row_major>& dst,
		tensor<float,dev_memory_space,row_major>& img,
		int factor,
		tensor<int,dev_memory_space,row_major>* indices) {
	int numImages = img.shape()[0];
	int imgPixels = img.shape()[1];
	int dstPixels = imgPixels * (factor * factor);
	int imgSize = sqrt(img.shape()[1]);
	int dstSize = imgSize * factor;
	
	cuvAssert(dstSize / factor == imgSize);
	
	cuvAssert(img.shape()[1]  *factor * factor == dst.shape()[1]);
	cuvAssert(img.shape()[0]== dst.shape()[0]);

	NVMatrix nv_img(img.ptr(), numImages, imgPixels, false);
	NVMatrix nv_dst(dst.ptr(), numImages, dstPixels, false);

	if(indices == NULL) {
		supersample(&nv_img, &nv_dst, factor);
	} else {
		assert(imgSize < 512);
		fill(dst, 0.0f);

		dim3 grid(imgSize, img.shape()[0]);
		dim3 threads(min(imgSize, 512));
		supersample_kernel<<<grid,threads>>>(dst.ptr(), img.ptr(), indices->ptr(), dstSize, factor, imgSize);
	}

	cuvSafeCall(cudaThreadSynchronize());
}

template<>
void supersample(tensor<float,host_memory_space,row_major>& dst,
		tensor<float,host_memory_space,row_major>& img,
		int factor,
		tensor<int,host_memory_space,row_major>* indices) {
	int numImages = img.shape()[0];
	int imgSize = sqrt(img.shape()[1]);
	int dstSize = imgSize * factor;

	cuvAssert(img.shape()[1]  *factor * factor == dst.shape()[1]);
	cuvAssert(img.shape()[0]== dst.shape()[0]);
	cuvAssert(dstSize / factor == imgSize);

	float* image = img.ptr();
	float* target = dst.ptr();

	if(indices != NULL) {
		for(int i = 0; i < numImages; i++) {
			for(int r = 0; r < imgSize; r++)
				for(int c = 0; c < imgSize; c++) {
					int idx = (*indices)[r*imgSize+c + i*imgSize*imgSize];
					int row = idx % factor;
					int col = idx / factor;
					target[(r*factor+row)*dstSize + c*factor+col] = image[r*imgSize + c];
				}
			target += dst.shape()[1];
			image += img.shape()[1];
		}
	} else {
		for(int i = 0; i < numImages; i++) {
			for(int r = 0; r < dstSize; r++)
				for(int c = 0; c < dstSize; c++) {
					target[0] = image[(r/factor)*imgSize+c/factor];
					target++;
				}
			image += img.shape()[1];
		}
	}

}

// arbitrary value can be anything <= 64 KB
#define CONST_SIZE 8192
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
void super_to_max(tensor<float,dev_memory_space,row_major>& dst,
		tensor<float,dev_memory_space,row_major>& img,
		int poolSize,
		int overlap,
		tensor<int,dev_memory_space,row_major>* indices,
		tensor<float,dev_memory_space,row_major>* filter) {
	cuvAssert(indices->shape()[1] == img.shape()[1]);
	cuvAssert(indices->shape()[0] == img.shape()[0]);
	cuvAssert(poolSize > overlap);
	int numImages = dst.shape()[0];
	cuvAssert(numImages == img.shape()[0]);
	int imgSize = sqrt(img.shape()[1]);
	if(imgSize * imgSize != img.shape()[1]){
			cout << std::endl<<"Error: imgSize x imgSize (" <<imgSize<<")²="<< imgSize*imgSize<<" should be img.w = "<<img.shape()[1]<<std::endl;
		}
	cuvAssert(imgSize * imgSize == img.shape()[1]);
	int stepSize = poolSize - overlap;
	int dstSize = (imgSize - 1) * stepSize + poolSize;
	if(dstSize * dstSize != dst.shape()[1]){
				cout << std::endl<<"Error: dstSize x dstSize (" <<dstSize<<")²="<< dstSize*dstSize<<" should be dst.w = "<<dst.shape()[1]<<std::endl;
			}
	cuvAssert(dstSize * dstSize == dst.shape()[1]);
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
		cuvAssert(filter->shape()[1] == poolSize);
		cuvAssert(filter->shape()[0] == poolSize);
		cuvAssert(sizeof(float) * filter->size() <= CONST_SIZE);
		cuvSafeCall( cudaMemcpyToSymbol(c_filter, filter->ptr(), sizeof(float) * filter->size(), 0, cudaMemcpyDeviceToDevice) );
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
void super_to_max(tensor<float,host_memory_space,row_major>& dst,
		tensor<float,host_memory_space,row_major>& img,
		int poolSize,
		int overlap,
		tensor<int,host_memory_space,row_major>* indices,
		tensor<float,host_memory_space,row_major>* filter) {
	cuvAssert(poolSize > overlap);
	int numImages = dst.shape()[0];
	cuvAssert(numImages == img.shape()[0]);
	int imgSize = sqrt(img.shape()[1]);
	cuvAssert(imgSize * imgSize == img.shape()[1]);
	int stepSize = poolSize - overlap;
	int dstSize = (imgSize - 1) * stepSize + poolSize;
	cuvAssert(dstSize * dstSize == dst.shape()[1]);
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
	void copy_into(tensor<float,dev_memory_space,row_major>& dst,
			  tensor<float,dev_memory_space,row_major>&   img,
			  int padding) {
	int inputSize = sqrt(img.shape()[1]);
	int outputSize = sqrt(dst.shape()[1]);
	cuvAssert(inputSize * inputSize == img.shape()[1]);
	cuvAssert(outputSize * outputSize == dst.shape()[1]);
	cuvAssert(inputSize + 2 * padding == outputSize);
	cuvAssert(img.shape()[0] == dst.shape()[0]);

	// make NVMatrices with this data
	NVMatrix nv_dst(dst.ptr(), dst.shape()[0], dst.shape()[1], false);
	NVMatrix nv_img(img.ptr(), img.shape()[0], img.shape()[1], false);

	copyInto(&nv_img, &nv_dst, padding, false);
}

template<>
	void copy_into(tensor<float,host_memory_space,row_major>& dst,
			  tensor<float,host_memory_space,row_major>&   img,
			  int padding) {
	int inputSize = sqrt(img.shape()[1]);
	int outputSize = sqrt(dst.shape()[1]);
	cuvAssert(inputSize * inputSize == img.shape()[1]);
	cuvAssert(outputSize * outputSize == dst.shape()[1]);
	cuvAssert(inputSize + 2 * padding == outputSize);
	cuvAssert(img.shape()[0] == dst.shape()[0]);

	float* img_ptr = img.ptr();
	float* dst_ptr = dst.ptr();
	for(int i=0; i<img.shape()[0]; i++) {
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
	void max_pooling(tensor<float,host_memory_space,row_major>& dst,
			tensor<float,host_memory_space,row_major>& img,
			unsigned int poolSize,
			unsigned int overlap,
			tensor<int,host_memory_space,row_major>* indices,
			tensor<float,host_memory_space,row_major>* filter) {
	if (indices!=NULL) {
		cuvAssert(indices->shape()[1] == dst.shape()[1]);
		cuvAssert(indices->shape()[0] == dst.shape()[0]);
	}

	cuvAssert(poolSize > overlap);
	int numImages = dst.shape()[0];
	cuvAssert(numImages == img.shape()[0]);
	int imgSize = sqrt(img.shape()[1]);
	cuvAssert(imgSize * imgSize == img.shape()[1]);
	int stepSize = poolSize - overlap;
	int dstSize = (imgSize - poolSize)/stepSize + 1;
	cuvAssert(dstSize * dstSize == dst.shape()[1]);
	cuvAssert((dstSize-1)*stepSize + poolSize == imgSize);
	if(filter!=NULL) {
		cuvAssert(filter->shape()[1] == poolSize);
		cuvAssert(filter->shape()[0] == poolSize);
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
					(*indices)(p, r*dstSize+c)=imax;
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

__global__
void first_pooling_zeros_kernel(float* img, int imgSize, int stepSize) {
	int tx = threadIdx.x; // ty = threadIdx.y;
	int bx = blockIdx.x, by = blockIdx.y;

	int p = tx + by * 256;
	if(p >= imgSize * imgSize)
		return;

	img += bx * imgSize * imgSize;

	int column = p % imgSize;
	int row = p / imgSize;

	// write result
	if ((column  % stepSize) || (row % stepSize)){
		img +=  p;
		*img = 0;
		}	
}

template<>
	void first_pooling_zeros(tensor<float,dev_memory_space,row_major>& img,
			unsigned int poolSize
			) {

	int numImages = img.shape()[0];
	int imgSize = sqrt(img.shape()[1]);
	cuvAssert(imgSize * imgSize == img.shape()[1]);
	int stepSize = poolSize;
	int dstSize = (imgSize - poolSize)/stepSize + 1;
	cuvAssert((dstSize-1)*stepSize + poolSize == imgSize);

	int numThreads = 256;
	int numBlocksX = numImages;
	int numBlocksY = ceil((float) (imgSize * imgSize)/numThreads);

	dim3 grid(numBlocksX, numBlocksY);
	dim3 threads(numThreads);
	first_pooling_zeros_kernel<<<grid,threads>>>(img.ptr(), imgSize, stepSize);

	cuvSafeCall(cudaThreadSynchronize());
}

__global__
void first_pooling_kernel(float* dst, float* img, int imgSize, int dstSize, int stepSize) {
	int tx = threadIdx.x; // ty = threadIdx.y;
	int bx = blockIdx.x, by = blockIdx.y;

	int p = tx + by * 256;
	if(p >= dstSize * dstSize)
		return;

	img += bx * imgSize * imgSize;

	int column = p % dstSize;
	int row = p / dstSize;

	// write result
	dst += bx * dstSize * dstSize + p;
	*dst = img[column*stepSize + (row*stepSize)*imgSize];
}

template<>
	void first_pooling(tensor<float,dev_memory_space,row_major>& dst,
			tensor<float,dev_memory_space,row_major>& img,
			unsigned int poolSize
			) {

	int numImages = dst.shape()[0];
	cuvAssert(numImages == img.shape()[0]);
	int imgSize = sqrt(img.shape()[1]);
	cuvAssert(imgSize * imgSize == img.shape()[1]);
	int stepSize = poolSize;
	int dstSize = (imgSize - poolSize)/stepSize + 1;
	cuvAssert(dstSize * dstSize == dst.shape()[1]);
	cuvAssert((dstSize-1)*stepSize + poolSize == imgSize);

	int numThreads = 256;
	int numBlocksX = numImages;
	int numBlocksY = ceil((float) (dstSize * dstSize)/numThreads);

	dim3 grid(numBlocksX, numBlocksY);
	dim3 threads(numThreads);
	first_pooling_kernel<<<grid,threads>>>(dst.ptr(), img.ptr(), imgSize, dstSize, stepSize);

	cuvSafeCall(cudaThreadSynchronize());
}

template<>
	void max_pooling(tensor<float,dev_memory_space,row_major>& dst,
			tensor<float,dev_memory_space,row_major>& img,
			unsigned int poolSize,
			unsigned int overlap,
			tensor<int,dev_memory_space,row_major>* indices,
			tensor<float,dev_memory_space,row_major>* filter) {

	if (indices!=NULL) {
		cuvAssert(indices->shape()[1] == dst.shape()[1]);
		cuvAssert(indices->shape()[0] == dst.shape()[0]);
	}
	cuvAssert(poolSize > overlap);
	int numImages = dst.shape()[0];
	cuvAssert(numImages == img.shape()[0]);
	int imgSize = sqrt(img.shape()[1]);
	cuvAssert(imgSize * imgSize == img.shape()[1]);
	int stepSize = poolSize - overlap;
	int dstSize = (imgSize - poolSize)/stepSize + 1;
	cuvAssert(dstSize * dstSize == dst.shape()[1]);
	cuvAssert((dstSize-1)*stepSize + poolSize == imgSize);

	int numThreads = 256;
	int numBlocksX = numImages;
	int numBlocksY = ceil((float) (dstSize * dstSize)/numThreads);

	if(filter!=NULL) {
		cuvAssert(filter->shape()[1] == poolSize);
		cuvAssert(filter->shape()[0] == poolSize);
		cuvAssert(sizeof(float) * filter->size() <= CONST_SIZE);
		cuvSafeCall( cudaMemcpyToSymbol(c_filter, filter->ptr(), sizeof(float) * filter->size(), 0, cudaMemcpyDeviceToDevice) );
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
	void strip_padding(tensor<float,dev_memory_space,row_major>& dst,
					   tensor<float,dev_memory_space,row_major>& img,
					   unsigned int padding) {
	int inputSize = sqrt(img.shape()[1]);
	int imgWidth = inputSize;
	int outputSize = sqrt(dst.shape()[1]);
	int numImages = img.shape()[0];
	cuvAssert(inputSize * inputSize == img.shape()[1]);
	cuvAssert(outputSize * outputSize == dst.shape()[1]);
	cuvAssert(inputSize - 2 * padding == outputSize);
	cuvAssert(img.shape()[0] == dst.shape()[0]);
	int numThreads = 256;
	int numBlocksX = ceil((float) (imgWidth * imgWidth)/numThreads);
	int numBlocksY = numImages;
	dim3 grid(numBlocksX, numBlocksY);
	dim3 dimBlock(numThreads,1);
	strip_padding_kernel<<<grid,dimBlock>>>(dst.ptr(), img.ptr(), imgWidth, numImages, padding);
	cuvSafeCall(cudaThreadSynchronize());
}

template<>
	void strip_padding(tensor<float,host_memory_space,row_major>& dst,
					   tensor<float,host_memory_space,row_major>& img,
					   unsigned int padding) {
	int inputSize = sqrt(img.shape()[1]);
	int imgWidth = inputSize;
	int outputSize = sqrt(dst.shape()[1]);
	int numImages = img.shape()[0];
	cuvAssert(inputSize * inputSize == img.shape()[1]);
	cuvAssert(outputSize * outputSize == dst.shape()[1]);
	cuvAssert(inputSize - 2 * padding == outputSize);
	cuvAssert(img.shape()[0] == dst.shape()[0]);
	fill(dst, 0.0f);


	int x,y, idx, idx_padded;
	float val;
	int stripped_width = imgWidth - 2 * padding;

	for (int imgIdx = 0; imgIdx < img.shape()[0]; imgIdx++){
		for(int px = 0; px < img.shape()[1]; px++){
			x = px % inputSize;
			y = px / inputSize;
			if ( x >=padding && x < padding+stripped_width &&
				 y >=padding && y < padding+stripped_width)
			{
				idx 		=	y*inputSize+x;
				idx_padded 	=	(y-padding)*stripped_width+(x-padding);
				val = img(imgIdx, idx);
				dst(imgIdx, idx_padded)=val;
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
	void row_ncopy(tensor<float,dev_memory_space,row_major>& dst,
				   tensor<float,dev_memory_space>& row,
				   unsigned int n) {
	int inputSize = row.size();
	cuvAssert(n == dst.shape()[0]);
	cuvAssert(n <= 4096);
	fill(dst, 0.0f);

	int numThreads = 256;
	int numBlocksX = ceil((float)inputSize/numThreads);
	int numBlocksY = 1;
	dim3 grid(numBlocksX, numBlocksY);
	dim3 dimBlock(numThreads,1);
	row_ncopy_kernel<<<grid,dimBlock>>>(dst.ptr(), row.ptr(), inputSize, n);
	cuvSafeCall(cudaThreadSynchronize());
}


template<>
	void row_ncopy(tensor<float,host_memory_space,row_major>& erg_h,
				   tensor<float,host_memory_space>& row,
				   unsigned int n) {

	cuvAssert(n == erg_h.shape()[0]);
	cuvAssert(n <= 4096);
	fill(erg_h, 0.0f);

	fill(erg_h, 0.0f);
	for(int idx = 0; idx < erg_h.shape()[1]; idx++ ){
		for (int idy = 0; idy < n; idy++){
			erg_h(idy,idx)=row[idx];
		}
	}
}


__global__ void cols_ncopy_kernel(float* targets, float* cols, const int rowSize, const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    int newRowSize = rowSize * n;

    //check if index is still in matrix
    if (idx < rowSize) {
	int offset_src_adr = idx + rowSize * row;
    	int offset_dst_adr = idx + newRowSize * row;
    	for(int i=0; i < n ;i++){
    	     *(targets + offset_dst_adr + i * rowSize)= *(cols+offset_src_adr);
    	}
    }
}


template<>
void cols_ncopy(	tensor<float, dev_memory_space, row_major>& dst,
			tensor<float, dev_memory_space, row_major>& col,
			unsigned int n){
	int inputSize 	= col.shape()[1]*col.shape()[0];
	int row_size 	= col.shape()[1];
	
	cuvAssert(n <= 4096);
	cuvAssert(dst.shape()[1] == row_size*n)
	fill(dst, 0.0f);
	int numThreads = 512;
	int numBlocksX = ceil((float)row_size/numThreads);
	int numBlocksY = col.shape()[0];
	dim3 grid(numBlocksX, numBlocksY);
	dim3 dimBlock(numThreads,1);
	cols_ncopy_kernel<<<grid,dimBlock>>>(dst.ptr(), col.ptr(), row_size, n);
	cuvSafeCall(cudaThreadSynchronize());

}


template<>
void cols_ncopy(tensor<float, host_memory_space, row_major>& dst,
		tensor<float, host_memory_space, row_major>& col,
		unsigned int n){
	int inputSize 	= col.shape()[1]*col.shape()[0];
	int row_size 	= col.shape()[1];
	cuvAssert(n <= 4096);
	cuvAssert(dst.shape()[1] == row_size*n)
	fill(dst, 0.0f);
	for(int r = 0; r < col.shape()[0]; r++){
		for(int c = 0; c < col.shape()[1]; c++){
			for(int j = 0; j < n; j++){		
				*(dst.ptr() 	+ r * row_size * n  // shift to correct row using new row width
						+ c 		    // shift by column
						+ j * row_size)	    // shift by old row size to the new position 
						= *(col.ptr() + r * row_size + c); 		
			}
		}	
	}
}



template<>
	void filter_rotate(tensor<float,host_memory_space,row_major>& dst,
					    tensor<float,host_memory_space,row_major>& filter,
					    unsigned int fs){
		int f = filter.shape()[1] / fs;
		float* f_h_ptr = filter.ptr();
		int row_offset=0;
		int f_h_w = filter.shape()[1];
		int numCases = filter.shape()[0];

		// iterate on every filter in a row
		for(int filter = 0; filter < f*fs; filter = filter+fs){
			// iterate on every element of the filter
			for(int y = 0; y < fs; y++){
				// every filterrow
				for(int nC = 0; nC <numCases; nC++){
					row_offset = nC*f_h_w;
					*(dst.ptr()+row_offset+filter+y) = *(f_h_ptr+row_offset+(fs-1)+filter-y);
				}

			}
		}
}

/*
 * this is limited to 22 x 22 filter kernels yet
 */
__global__ void filter_rotate_kernel(float* dst, float* src, const int h, const int w, const int fs, const int size) {

	const int col_idx = threadIdx.x;
	const int row_idx = blockIdx.y;

	// load weights in shared memory
	__shared__  float filter[512];

	int px_adr_glob = 0;

	// check if col idx is less than the number of cells in one row and less than the number of cells at all (at bottom of matrix)
	if( (col_idx < w) && (row_idx * w + col_idx <= size)){
		// I. load pixels in a coalesced way

		// global memory adress for pixel
		px_adr_glob = row_idx * w + col_idx;

		//load filter element
		*(filter+col_idx) =  *(src+px_adr_glob);

		// wait until everything is loaded
		__syncthreads();

		// II. now write with hopefully only few bank conflicts
		int filter_start = (col_idx / fs) * fs;
		int filter_element_idx = col_idx % fs;

		*(dst+px_adr_glob) = *(filter + filter_start + (fs-1) - filter_element_idx);
	}

}

template<>
void filter_rotate(	tensor<float,dev_memory_space,row_major>& dst,
					tensor<float,dev_memory_space,row_major>& filter,
					unsigned int fs){
		cuvAssert(dst.shape()[0] == filter.shape()[0])
		cuvAssert(dst.shape()[1] == filter.shape()[1])

		int num_filter = filter.shape()[1] / fs;
		cuvAssert(sqrt(fs) <= 22)

		float* f_h_ptr = filter.ptr();
		int f_h_w = filter.shape()[1];
		int numCases = filter.shape()[0];

		// we put as many filter in a row of width 512 as possible
		int numFiltersPerRow = 512 / fs;
		int numRows = ceil((float)(num_filter*filter.shape()[0]) / numFiltersPerRow);
		//std::cout << "resizing from " << num_filter << "x" << filter.shape()[0] << " to " << numFiltersPerRow << " x " << numRows << std::endl;
		int _h = numRows;
		int _w = numFiltersPerRow*fs;

		int numThreads = 512;
		int numBlocksX = 1;
		int numBlocksY = _h;
		dim3 grid(numBlocksX, numBlocksY);
		dim3 dimBlock(numThreads,1);

//		std::cout << "filter.h =  " << filter.shape()[0] << std::endl;
		filter_rotate_kernel<<<grid,dimBlock>>>(dst.ptr(), filter.ptr(), _h, _w, fs, filter.shape()[0]*filter.shape()[1]);
		cuvSafeCall(cudaThreadSynchronize());

}

//__global__ void add_maps_h_kernel(float* dst, float* img, const int img_w, const int imagesize) {
//
//	int px = threadIdx.x +  blockDim.x * blockIdx.x;
//	int row = blockIdx.y;
//
//	int num_maps = img_w / imagesize;
//
//	__shared__ float summedMaps[512];
//
//	// sum up in fast shared mem
//	for(int i = 0; i < num_maps; i++){
//		summedMaps[px] += *(img + row * img_w		// goto row in matrix
//								+ px				// pixel
//								+ i * imagesize);   // iterate on images
//	}
//
//	// move result to global mem
//	*(dst + row * img_w + px) = *(summedMaps + row * img_w + px);
//}
//
//template<>
//void add_maps_h(	dense_matrix<float,dev_memory_space,row_major>& dst,
//					dense_matrix<float,dev_memory_space,row_major>& mat,
//					unsigned int image_size){
//
//		int num_images = mat.shape()[1] / image_size;
//		cuvAssert(dst.shape()[1] == image_size);
//		cuvAssert(dst.shape()[0] == mat.shape()[0]);
//		cuvAssert(num_images * image_size == mat.shape()[1]);
//
//		int numThreads = 512;
//		int numBlocksX = ceil((float)mat.shape()[1]/numThreads);
//		int numBlocksY = mat.shape()[0];
//		dim3 grid(numBlocksX, numBlocksY);
//		dim3 dimBlock(numThreads,1);
//
//		add_maps_h_kernel<<<grid,dimBlock>>>(dst.ptr(), mat.ptr(), mat.shape()[1], image_size);
//		cuvSafeCall(cudaThreadSynchronize());
//}
//
//template<>
//void add_maps_h(	dense_matrix<float,host_memory_space,row_major>& dst,
//					dense_matrix<float,host_memory_space,row_major>& mat,
//					unsigned int image_size){
//
//		int num_images = mat.shape()[1] / image_size;
//		cuvAssert(dst.shape()[1] == image_size);
//		cuvAssert(dst.shape()[0] == mat.shape()[0]);
//		cuvAssert(num_images * image_size == mat.shape()[1]);
//
//		float* e_ptr = dst.ptr();
//		float* i_ptr = mat.ptr();
//
//		// host solution
//		for (int row = 0; row<mat.shape()[0]; row++){
//			for(int px = 0; px < image_size; px++){
//				for(int img = 0; img < num_images; img++){
//					*(e_ptr + row*dst.shape()[1] + px) += *(i_ptr + row * dst.shape()[1]  // move to right row
//															 + img * image_size // move to img
//															 + px);				// move to pixel in img
//				}
//			}
//		}
//}

__global__ void calc_error_to_blob_kernel(float* img,
										  float* src,
										  float* blob,
										  const int img_w,
										  const int img_h,
										  float sigma,
										  const int num_maps,
										  float temporal_weight,
										  float interval_size,
										  float interval_offset,
										  const int window_radius) {

	int idx = threadIdx.x +  blockDim.x * blockIdx.x;
	int row = blockIdx.y;

	int x = idx % img_w;
	int y = idx / img_w;

	float center_x = *(blob+row*2);
	float center_y = *(blob+row*2+1);

	float a = (float)(x - center_x)/ sigma;
	float b = (float)(y - center_y)/ sigma;

	// destination is calculated by the row the pixel is in (row*imagesize) and the index in the picture (idx)
	// img_w and img_h refers to the dimensions of an image (one row) in the img matrix

	//p(x,α,σ) = 1/sqrt(2πσ²)*exp(-(x-α)²/2σ²)
	if(idx < img_w * img_h){
		if (window_radius > 0){
			if(		window_radius*window_radius > pow(a*sigma,2) + pow(a*sigma,2)
				or 	(x == center_x and y==center_y)){
				float gauss_value = interval_size*expf(-(a*a + b*b)/2.f)-interval_offset;
				float act_val = *(src+idx+row*(img_w*img_h));
				*(img+idx+row*(img_w*img_h)) =(temporal_weight * (gauss_value - act_val));
			}else
				*(img+idx+row*(img_w*img_h)) = 0.0f;
		}else{
			float gauss_value = interval_size*expf(-(a*a + b*b)/2.f)-interval_offset;
			float act_val = *(src+idx+row*(img_w*img_h));
			*(img+idx+row*(img_w*img_h)) =(temporal_weight * (gauss_value - act_val));
		}
	}
}



template<>
void calc_error_to_blob(	tensor<float,dev_memory_space,row_major>& dst,
							tensor<float,dev_memory_space,row_major>& img,
							tensor<float,dev_memory_space,row_major>& blob_mat,
							unsigned int image_w,
							unsigned int image_h,
							float sigma,
							float temporal_weight,
							float interval_size,
							float interval_offset,
							unsigned int window_radius){
	cuvAssert(dst.shape()[0] == img.shape()[0]);
	cuvAssert(dst.shape()[1] == img.shape()[1]);

	int numThreads = 512;
	int numBlocksX = ceil((float)img.shape()[1]/numThreads);
	int numBlocksY = dst.shape()[0];

	int numTeacherMaps = blob_mat.shape()[0];

	dim3 grid(numBlocksX, numBlocksY);
	dim3 dimBlock(numThreads,1);

	calc_error_to_blob_kernel<<<grid,dimBlock>>>(	dst.ptr(),
													img.ptr(),
													blob_mat.ptr(),
													image_w,
													image_h,
													sigma,
													numTeacherMaps,
													temporal_weight,													
													interval_size,
													interval_offset,
													window_radius);
	cuvSafeCall(cudaThreadSynchronize());
};

template<>
void calc_error_to_blob(	tensor<float,host_memory_space,row_major>& dst,
							tensor<float,host_memory_space,row_major>& img,
							tensor<float,host_memory_space,row_major>& blob_mat,
							unsigned int image_w,
							unsigned int image_h,
							float sigma,
							float temporal_weight,
							float interval_size,
							float interval_offset,
							unsigned int window_radius){
	cuvAssert(dst.shape()[0] == img.shape()[0]);
	cuvAssert(dst.shape()[1] == img.shape()[1]);

	float center_x 		= 0;
	float center_y	 	= 0;
	float a 			= 0;
	float b 			= 0;
	float gauss_value 	= 0;
	float act_val 		= 0;
	int idx 			= 0;
	int mapsize = image_w*image_h;

	for(int y=0; y < image_h; y++){
		for(int x=0; x < image_w; x++){
			for(int map = 0; map < blob_mat.shape()[0]; map++){
				center_x = *(blob_mat.ptr()+map*2);
				center_y = *(blob_mat.ptr()+map*2+1);

				a = (float)(x - center_x)/ sigma;
				b = (float)(y - center_y)/ sigma;

				idx = y * image_w + x;

				gauss_value = interval_size*expf(-(a*a + b*b)/2.f)-interval_offset;
				act_val = *(img.ptr()+idx+map*mapsize);
				*(dst.ptr()+idx+map*mapsize) =(temporal_weight * (gauss_value - act_val));
			}
		}
	}

};

__global__ void check_exitatory_inhibitory_kernel(float* filter,
												  const int filter_w,
												  const int filter_h,
												  const int start_filter,
												  const int filter_pixels,
												  const int num_inhibitory,
												  const int num_exitatory) {

	int idx = threadIdx.x +  blockDim.x * blockIdx.x;
	int row = blockIdx.y;

	int inhib_start_col 	= start_filter * filter_pixels;
	int exit_start_col		= inhib_start_col + num_inhibitory*filter_pixels;

	int ptr_adr =  	inhib_start_col		//move to the beginning of the block
						+ idx				//move to column in block
						+ (row*filter_w);	// move pointer by row many rows down

	if(idx < filter_w and row<filter_h)
		if(idx >= exit_start_col - inhib_start_col){ // if idx is in exitatory block
			if(*(filter+ptr_adr) < 0)
				*(filter + ptr_adr) = 0;
		}else{										 // if idx is in inhibitory
			if(*(filter + ptr_adr) > 0)
				*(filter + ptr_adr) = 0;
		}

}

template<>
void check_exitatory_inhibitory(tensor<float,dev_memory_space,row_major>& filter,
								unsigned int start_filter,
								unsigned int filter_pixels,
								unsigned int num_inhibitory,
								unsigned int num_exitatory){

	int row_size = filter.shape()[1];
	int inhib_start_col 	= start_filter * filter_pixels;
	int inhib_end_col 		= inhib_start_col + num_inhibitory * filter_pixels - 1;
	int exit_start_col		= inhib_end_col + 1;
	int exit_end_col		= exit_start_col + num_exitatory * filter_pixels - 1;
//	std::cout << "filter.h: " << filter.shape()[0] << " cols ges: "<< filter.shape()[1] <<"inhib_start: " << inhib_start_col << " inhib_end: " << inhib_end_col << " exhib_start: "<< exit_start_col << " exit_end: "<< exit_end_col<< std::endl;
	int numThreads = 512;
	int numBlocksX = ceil((float)(exit_end_col-inhib_start_col)/numThreads);
 	int numBlocksY = filter.shape()[0];
// 	std::cout << "launching " << numBlocksX << "x" << numBlocksY << "x512 Threads for " << (exit_end_col-inhib_start_col)*filter.shape()[0] <<" elements"<<std::endl;
	dim3 grid(numBlocksX, numBlocksY);
	dim3 dimBlock(numThreads,1);

	check_exitatory_inhibitory_kernel<<<grid,dimBlock>>>( filter.ptr(),
														  filter.shape()[1],
														  filter.shape()[0],
														  start_filter,
														  filter_pixels,
														  num_inhibitory,
														  num_exitatory);
	cuvSafeCall(cudaThreadSynchronize());

};

template<>
void check_exitatory_inhibitory(tensor<float,host_memory_space,row_major>& filter,
								unsigned int start_filter,
								unsigned int filter_pixels,
								unsigned int num_inhibitory,
								unsigned int num_exitatory){

	int row_size = filter.shape()[1];
	int inhib_start_col 	= start_filter * filter_pixels;
	int inhib_end_col 		= inhib_start_col + num_inhibitory * filter_pixels-1;
	int exit_start_col		= inhib_end_col+1;
	int exit_end_col		= exit_start_col + num_exitatory * filter_pixels-1;
	std::cout << "filter.h: " << filter.shape()[0] << " cols ges: "<< filter.shape()[1] <<"inhib_start: " << inhib_start_col << " inhib_end: " << inhib_end_col << " exhib_start: "<< exit_start_col << " exit_end: "<< exit_end_col<< std::endl;

	// horizontal direction
	for(int c = inhib_start_col; c < inhib_end_col; c = c + 1 ){
		// vertical direction
		for(int r = 0; r < filter.shape()[0]; r++){
			if(*(filter.ptr()+c+(r*row_size)) > 0)
				*(filter.ptr()+c+(r*row_size)) = 0;
		}
	}

	for(int c = exit_start_col; c <= exit_end_col; c = c + 1 ){
		// vertical direction
		for(int r = 0; r < filter.shape()[0]; r++){
			if(*(filter.ptr()+c+(r*row_size)) < 0)
				*(filter.ptr()+c+(r*row_size)) = 0;
		}
	}

};

__global__ void init_exitatory_inhibitory_kernel(float* filter,
												  const int filter_w,
												  const int filter_h,
												  const int start_filter,
												  const int filter_pixels,
												  const int num_inhibitory,
												  const int num_exitatory) {

	int idx = threadIdx.x +  blockDim.x * blockIdx.x;
	int row = blockIdx.y;

	int inhib_start_col 	= start_filter * filter_pixels;
	int exit_start_col		= inhib_start_col + num_inhibitory*filter_pixels;

	int ptr_adr =  	inhib_start_col		//move to the beginning of the block
						+ idx				//move to column in block
						+ (row*filter_w);	// move pointer by row many rows down

	if(idx < filter_w and row<filter_h)
		if(idx >= exit_start_col - inhib_start_col){ // if idx is in exitatory block
			if(*(filter+ptr_adr) < 0)
				*(filter + ptr_adr) = -1 * *(filter + ptr_adr);
		}else{										 // if idx is in inhibitory
			if(*(filter + ptr_adr) > 0)
				*(filter + ptr_adr) = -1 * *(filter + ptr_adr);
		}

}

template<>
void init_exitatory_inhibitory(tensor<float,dev_memory_space,row_major>& filter,
								unsigned int start_filter,
								unsigned int filter_pixels,
								unsigned int num_inhibitory,
								unsigned int num_exitatory){

	int row_size = filter.shape()[1];
	int inhib_start_col 	= start_filter * filter_pixels;
	int inhib_end_col 		= inhib_start_col + num_inhibitory * filter_pixels - 1;
	int exit_start_col		= inhib_end_col + 1;
	int exit_end_col		= exit_start_col + num_exitatory * filter_pixels - 1;
//	std::cout << "filter.h: " << filter.shape()[0] << " cols ges: "<< filter.shape()[1] <<"inhib_start: " << inhib_start_col << " inhib_end: " << inhib_end_col << " exhib_start: "<< exit_start_col << " exit_end: "<< exit_end_col<< std::endl;
	int numThreads = 512;
	int numBlocksX = ceil((float)(exit_end_col-inhib_start_col)/numThreads);
 	int numBlocksY = filter.shape()[0];
// 	std::cout << "launching " << numBlocksX << "x" << numBlocksY << "x512 Threads for " << (exit_end_col-inhib_start_col)*filter.shape()[0] <<" elements"<<std::endl;
	dim3 grid(numBlocksX, numBlocksY);
	dim3 dimBlock(numThreads,1);

	check_exitatory_inhibitory_kernel<<<grid,dimBlock>>>( filter.ptr(),
														  filter.shape()[1],
														  filter.shape()[0],
														  start_filter,
														  filter_pixels,
														  num_inhibitory,
														  num_exitatory);
	cuvSafeCall(cudaThreadSynchronize());

};

template<>
void init_exitatory_inhibitory(tensor<float,host_memory_space,row_major>& filter,
								unsigned int start_filter,
								unsigned int filter_pixels,
								unsigned int num_inhibitory,
								unsigned int num_exitatory){

	int row_size = filter.shape()[1];
	int inhib_start_col 	= start_filter * filter_pixels;
	int inhib_end_col 		= inhib_start_col + num_inhibitory * filter_pixels-1;
	int exit_start_col		= inhib_end_col+1;
	int exit_end_col		= exit_start_col + num_exitatory * filter_pixels-1;
	std::cout << "filter.h: " << filter.shape()[0] << " cols ges: "<< filter.shape()[1] <<"inhib_start: " << inhib_start_col << " inhib_end: " << inhib_end_col << " exhib_start: "<< exit_start_col << " exit_end: "<< exit_end_col<< std::endl;

	// horizontal direction
	for(int c = inhib_start_col; c < inhib_end_col; c = c + 1 ){
		// vertical direction
		for(int r = 0; r < filter.shape()[0]; r++){
			if(*(filter.ptr()+c+(r*row_size)) > 0)
				*(filter.ptr()+c+(r*row_size)) = -1 * *(filter.ptr()+c+(r*row_size));
		}
	}

	for(int c = exit_start_col; c <= exit_end_col; c = c + 1 ){
		// vertical direction
		for(int r = 0; r < filter.shape()[0]; r++){
			if(*(filter.ptr()+c+(r*row_size)) < 0)
				*(filter.ptr()+c+(r*row_size)) = -1 * *(filter.ptr()+c+(r*row_size));
		}
	}

};

}
