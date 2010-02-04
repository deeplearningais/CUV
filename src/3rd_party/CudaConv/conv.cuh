/*
 * CUDA 2.1-2.3 convolution routines optimized for GT200 architecture.
 *
 * Nov 1, 2009
 * Alex Krizhevsky (akrizhevsky@gmail.com)
 */
#ifndef CONV_CUH_
#define CONV_CUH_

#include <cutil_inline.h>
#include <assert.h>
#include <matrix.h>
#include <nvmatrix.cuh>
#include "conv_common.cuh"

void convolve_bw(NVMatrix* images, NVMatrix* filters, NVMatrix* targets);
void convolve_color(NVMatrix* images, NVMatrix* filters, NVMatrix* targets);

/*
 * This version uses block size (z, y, x) = 8x4x16.
 *
 * Each block convolves 1 image with 16 filters.
 * Works for filters 14x14 or smaller; image size only influences checkBounds.
 * The checked version uses 20 registers...would be nice to get that down to 16.
 */
template<int filterSize, bool checkBounds, int stride>
__global__ void conv_bw_fit_4x16_2per(float* imgs, float* filters, float* targets, const int imgSize) {
    const int shImgSizeX = filterSize + 15, shImgSizeY = filterSize + 3;
    __shared__ float shImg[shImgSizeY][shImgSizeX];
    __shared__ float shFilter[16][filterSize][filterSize];

    const int imgIdx = blockIdx.x;
    const int filtIdx = 2 * 8 * blockIdx.y + 2 * threadIdx.z;
    const int numFilters = 2 * 8 * gridDim.y;
    const int pidx = threadIdx.y * 16 + threadIdx.x; // thread's index within the 4x16 "plate" of threads
    const int tidx = threadIdx.z * 4 * 16 + pidx; // thread's index within its block
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = MUL24(numOutputsX, numOutputsX);
    const int imgPixels = MUL24(imgSize, imgSize); // size of image

    const int shImgPixels = shImgSizeX * shImgSizeY; // size of shared buffer for image
    const int filterPixels = filterSize * filterSize;
    const int loadX = tidx % (shImgSizeX);
    const int loadY = tidx / (shImgSizeX);

    imgs += imgIdx * MUL24(stride, imgPixels) + MUL24(loadY, imgSize) + loadX;
    filters += filtIdx * MUL24(stride, filterPixels) + pidx;
    targets += imgIdx * numFilters * numOutputs + MUL24(filtIdx, numOutputs) + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;

    float* shFilterLoad = &shFilter[threadIdx.z * 2][0][pidx];
    float* shFilterLoad2 = &shFilter[threadIdx.z * 2 + 1][0][pidx];
    float* shImgLoad = &shImg[loadY][loadX];

    for (int i = pidx; i < filterPixels; i += 16 * 4) { // Load the filter
        shFilterLoad[0] = filters[0];
        shFilterLoad2[0] = filters[MUL24(filterPixels, stride)];
        filters += 16 * 4;
        shFilterLoad += 16 * 4;
        shFilterLoad2 += 16 * 4;
    }
//    const bool load = tidx < shImgPixels;
    for(int y = 0; y < numOutputsX; y += 4) {
        for(int x = 0; x < numOutputsX; x += 16) {
            __syncthreads();
            /*
             * This will load the entire (shared) image AS LONG AS THE FILTER SIZE IS <= 14.
             * If the filter size gets too big there won't be enough threads in the block
             * to load the entire shared image memory in one go. But that's ok because
             * when the filter size is > 14, this function doesn't have enough
             * memory to run anyway.
             *
             * O = (I - F + 1)
             * If O = 16K, then I = 15 + F + 16(K - 1). This is why !checkBounds is here.
             */
            if (tidx < shImgPixels) {
                if (!checkBounds || (x + loadX < imgSize && y + loadY < imgSize)) { // very very cheap test (~0.3% of runtime)
                    shImgLoad[0] = imgs[MUL24(y, imgSize) + x];
                }
            }

            __syncthreads();

            if (!checkBounds || (x + threadIdx.x < numOutputsX && y + threadIdx.y < numOutputsX)) {
                float* myShFilter = &shFilter[2 * threadIdx.z][0][0];
                float* myShImg = &shImg[threadIdx.y][threadIdx.x];
                float prod[2] = { 0, 0 };
                // The checkBounds version sees a slight speedup here
                // from #parama unroll, but the nocheck version sees a slowdown (because occupancy goes down).
                #pragma unroll
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        prod[0] += myShFilter[0] * myShImg[0];
                        prod[1] += myShFilter[filterPixels] * myShImg[0];

                        myShFilter++;
                        myShImg++;
                    }
                    myShImg += 15;
                }

                if (stride == 1) {
                    targets[MUL24(y, numOutputsX) + x] += prod[0];
                    targets[MUL24(y, numOutputsX) + x + numOutputs] += prod[1];
                } else {
                    targets[MUL24(y, numOutputsX) + x] += prod[0];
                    targets[MUL24(y, numOutputsX) + x + numOutputs] += prod[1];
                }
            }
        }
    }
}


/*
 * This version uses block size (z, y, x).
 *
 * Each block convolves 1 image with 8 filters.
 * Works for filters 20x20 or smaller; image size only influences checkBounds.
 */
template<int filterSize, bool checkBounds, int stride>
__global__ void conv_bw_fit_4x16_1per(float* imgs, float* filters, float* targets, int imgSize) {
    const int shImgSizeX = filterSize + 15, shImgSizeY = filterSize + 3;
    const int shImgPixels = shImgSizeX * shImgSizeY; // size of shared buffer for image
    __shared__ float shImg[shImgSizeY][shImgSizeX];
    __shared__ float shFilter[8][filterSize][filterSize];

    const int imgIdx = blockIdx.x;
    const int filtIdx = 8 * blockIdx.y + threadIdx.z;
    const int numFilters = 8 * gridDim.y;
    const int pidx = threadIdx.y * 16 + threadIdx.x; // thread's index within the 4x16 "plate" of threads
    const int tidx = threadIdx.z * 4 * 16 + pidx; // thread's index within its block
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = MUL24(numOutputsX, numOutputsX);
    const int imgPixels = MUL24(imgSize, imgSize); // size of image
    const int filterPixels = filterSize * filterSize;

    imgs += imgPixels * MUL24(imgIdx, stride);
    filters += filtIdx * MUL24(filterPixels, stride) + pidx;
    targets += imgIdx * numFilters * numOutputs + MUL24(filtIdx, numOutputs) + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;

    float* shFilterLoad = &shFilter[threadIdx.z][0][pidx];

    for (int i = pidx; i < filterPixels; i += 16 * 4) { // Load the filter
        shFilterLoad[0] = filters[0];
        filters += 16 * 4;
        shFilterLoad += 16 * 4;
    }

    for(int y = 0; y < numOutputsX; y += 4) {
        for(int x = 0; x < numOutputsX; x += 16) {
            __syncthreads();
            for (int i = tidx; i < shImgPixels; i += 16 * 4 * 8) {
                const int loadX = i % (shImgSizeX);
                const int loadY = i / (shImgSizeX);
                // TODO; don't need loadY, loadX to index shImg. can use i
                if (!checkBounds || (x + loadX < imgSize && y + loadY < imgSize)) {
                    shImg[0][i] = imgs[(loadY + y) * imgSize + loadX + x];
                }
            }

            __syncthreads();

            if (!checkBounds || (x + threadIdx.x < numOutputsX && y + threadIdx.y < numOutputsX)) {
                float* myShFilter = &shFilter[threadIdx.z][0][0];
                float* myShImg = &shImg[threadIdx.y][threadIdx.x];
                float prod = 0;

                #pragma unroll
                for (int i = 0; i < filterSize; i++) {
//                    #pragma unroll
                    for (int j = 0; j < filterSize; j++) {
                        prod += myShFilter[0] * myShImg[0];

                        myShFilter++;
                        myShImg++;
                    }
                    myShImg += 15;
                }
//                if(stride == 1) {
//                    targets[0] = prod;
//                } else {
//                    targets[0] += prod;
//                }
                if(stride == 1) {
                    targets[MUL24(y, numOutputsX) + x] += prod;
                } else {
                    targets[MUL24(y, numOutputsX) + x] += prod;
                }
            }
//            targets += !checkBounds || x < numOutputsX - 16 ? 16 : numOutputsX - x;
        }
//        targets += MUL24(3, numOutputsX);
    }
}

/*
 * This version uses block size (z, y, x) = 8x4x16.
 *
 * Each block convolves 1 image with 16 filters.
 * Use only when the filter size is > 14, otherwise use the functions that
 * cache the entire filter.
 */
template<bool checkFilterBounds, int stride>
__global__ void conv_bw_nofit_4x16_2per(float* imgs, float* filters, float* targets, int imgSize, int filterSize) {
    const int shImgSizeX = 16 + 15, shImgSizeY = 4 + 3;
    __shared__ float shImg[shImgSizeY][shImgSizeX];
    __shared__ float shFilter[16][4][16];

    const int imgIdx = blockIdx.x;
    const int filtIdx = 2 * 8 * blockIdx.y + 2 * threadIdx.z;
    const int numFilters = 2 * 8 * gridDim.y;
    const int pidx = threadIdx.y * 16 + threadIdx.x; // thread's index within the 4x16 "plate" of threads
    const int tidx = threadIdx.z * 4 * 16 + pidx; // thread's index within its block
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = MUL24(numOutputsX, numOutputsX);
    const int imgPixels = MUL24(imgSize, imgSize); // size of image

    const int shImgPixels = shImgSizeY * shImgSizeX; // size of shared buffer for image
    const int filterPixels = MUL24(filterSize, filterSize);
    const int loadX = tidx % (shImgSizeX);
    const int loadY = tidx / (shImgSizeX);
    const int filterStride =  MUL24(filterPixels, stride);

    imgs += MUL24(MUL24(imgPixels, stride), imgIdx) + MUL24(loadY, imgSize) + loadX;
    filters += MUL24(filtIdx, filterStride) + MUL24(threadIdx.y, filterSize) + threadIdx.x;
    targets += MUL24(imgIdx, MUL24(numFilters, numOutputs)) + MUL24(filtIdx, numOutputs) + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;

    float* shFilterLoad = &shFilter[threadIdx.z * 2][0][pidx];
    float* shFilterLoad2 = &shFilter[threadIdx.z * 2 + 1][0][pidx];
    float* shImgLoad = &shImg[loadY][loadX];

    for(int y = 0; y < numOutputsX; y += 4) {
        for (int x = 0; x < numOutputsX; x += 16) {
            float prod[2] = { 0, 0 };
            const bool compute = (x + threadIdx.x < numOutputsX && y + threadIdx.y < numOutputsX);
            for (int fY = 0; fY < filterSize; fY += 4) {
                for (int fX = 0; fX < filterSize; fX += 16) {

                    __syncthreads();
                    shFilterLoad[0] = 0;
                    shFilterLoad2[0] = 0;
                    if (!checkFilterBounds || (threadIdx.x + fX < filterSize && threadIdx.y + fY < filterSize)) {
                        float* f = &filters[MUL24(fY, filterSize) + fX];
                        shFilterLoad[0] = f[0];
                        shFilterLoad2[0] = f[filterStride];
                    }
//                    filters += !checkFilterBounds || fX < filterSize - 16 ? 16 : filterSize - fX;

                    if (tidx < shImgPixels && (x + fX + loadX < imgSize && y + fY + loadY < imgSize)) {
                        // I tried incrementing imgs instead of indexing it, but that
                        // uses more registers and doesn't speed things up much.
                        shImgLoad[0] = imgs[MUL24((y + fY), imgSize) + x + fX];
                    }

                    __syncthreads();

                    if (compute) {
                        float* myShFilter = &shFilter[2 * threadIdx.z][0][0];
                        float* myShImg = &shImg[threadIdx.y][threadIdx.x];
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            #pragma unroll
                            for (int j = 0; j < 16; j++) {
                                prod[0] += myShFilter[0] * myShImg[0];
                                prod[1] += myShFilter[16 * 4] * myShImg[0];

                                myShFilter++;
                                myShImg++;
                            }
                            myShImg += 15;
                        }
                    }
                }
            }
            if (compute) {
                if (stride == 1) {
                    targets[MUL24(y, numOutputsX) + x] += prod[0];
                    targets[MUL24(y, numOutputsX) + x + numOutputs] += prod[1];
                } else {
                    targets[MUL24(y, numOutputsX) + x] += prod[0];
                    targets[MUL24(y, numOutputsX) + x + numOutputs] += prod[1];
                }
            }
        }
    }
}
#endif /* CONV_CUH_ */
