//*LB*
// Copyright (c) 2009, Alexander Krizhevsky
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
//  * Neither the name of the University of Toronto 
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




/*
 * conv_extras.cuh
 *
 *  Created on: Nov 10, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#ifndef CONV_EXTRAS_CUH_
#define CONV_EXTRAS_CUH_
#include "cutil_inline.h"

#define MUL24 __mul24
/*
 * This version uses a fixed block size of (z, y, x) = 8x4x16,
 * but the filter cache size is a compile-time parameter, so it can
 * be set to something that divides, or almost divides, the filter size to avoid
 * wasted iterations.
 *
 * FILTER CACHE SIZE MUST DIVIDE FILTER SIZE.
 *
 * Each block convolves 1 image with 16 filters.
 * Use only when the filter size is > 14, otherwise use the functions that
 * cache the entire filter.
 */
template<bool checkOutputBounds, int shFilterSizeY, int shFilterSizeX, int stride>
__global__ void conv_bw_nofit_4x16_dynfilter_2per(float* imgs, float* filters, float* targets, int imgSize, int filterSize) {
    const int shImgSizeX = 16 + shFilterSizeX - 1, shImgSizeY = 4 + shFilterSizeY - 1;
    __shared__ float shImg[shImgSizeY][shImgSizeX];
    __shared__ float shFilter[16][shFilterSizeY][shFilterSizeX];

    const int imgIdx = blockIdx.x;
    const int filtIdx = 2 * 8 * blockIdx.y + 2 * threadIdx.z;
    const int numFilters = 2 * 8 * gridDim.y;
    const int pidx = threadIdx.y * 16 + threadIdx.x; // thread's index within the 4x16 "plate" of threads
    const int tidx = threadIdx.z * 4 * 16 + pidx; // thread's index within its block
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = MUL24(numOutputsX, numOutputsX);
    const int imgPixels = MUL24(imgSize, imgSize); // size of image

    const int shFilterPixels = shFilterSizeX*shFilterSizeY;
    const int shImgPixels = MUL24(shImgSizeY, shImgSizeX); // size of shared buffer for image
    const int filterPixels = MUL24(filterSize, filterSize);
//    const int loadX = tidx % (shImgSizeX);
//    const int loadY = tidx / (shImgSizeX);
    const int filterStride =  MUL24(filterPixels, stride);

    imgs += MUL24(MUL24(imgPixels, stride), imgIdx);
    filters += MUL24(filtIdx, filterStride);
    targets += MUL24(imgIdx, MUL24(numFilters, numOutputs)) + MUL24(filtIdx, numOutputs) + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;

    float* shFilterLoad = &shFilter[threadIdx.z * 2][0][0];
//    float* shFilterLoad2 = &shFilter[threadIdx.z * 2 + 1][0][0];
//    float* shImgLoad = &shImg[loadY][loadX];

    for(int y = 0; y < numOutputsX; y += 4) {
        for (int x = 0; x < numOutputsX; x += 16) {
            float prod[2] = { 0, 0 };
            const bool compute = !checkOutputBounds || (x + threadIdx.x < numOutputsX && y + threadIdx.y < numOutputsX);
            for (int fY = 0; fY < filterSize; fY += shFilterSizeY) {
                for (int fX = 0; fX < filterSize; fX += shFilterSizeX) {

                    __syncthreads();

                    for (int i = pidx; i < shFilterPixels; i += 16 * 4) {
                        const int loadX = i % (shFilterSizeX);
                        const int loadY = i / (shFilterSizeX);
                        float* a = &filters[MUL24(fY + loadY, filterSize) + fX + loadX];
                        shFilterLoad[i] = a[0];
                        shFilterLoad[i + shFilterPixels] = a[filterStride];
                    }

//                    filters += !checkFilterBounds || fX < filterSize - 16 ? 16 : filterSize - fX;

                    for(int i = tidx; i < shImgPixels; i += 16*4*8) {
                        const int loadX = i % (shImgSizeX);
                        const int loadY = i / (shImgSizeX);
                        if (!checkOutputBounds || (x + fX + loadX < imgSize && y + fY + loadY < imgSize)) {
                            // I tried incrementing imgs instead of indexing it, but that
                            // uses more registers and doesn't speed things up much.
                            shImg[0][i] = imgs[MUL24((y + fY + loadY), imgSize) + x + fX + loadX];
                        }
                    }

                    __syncthreads();

                    if (compute) {
                        float* myShFilter = &shFilter[2 * threadIdx.z][0][0];
                        float* myShImg = &shImg[threadIdx.y][threadIdx.x];
                        #pragma unroll
                        for (int i = 0; i < shFilterSizeY; i++) {
//                            #pragma unroll
                            for (int j = 0; j < shFilterSizeX; j++) {
                                prod[0] += myShFilter[0] * myShImg[0];
                                prod[1] += myShFilter[shFilterPixels] * myShImg[0];

                                myShFilter++;
                                myShImg++;
                            }
                            myShImg += 15;
                        }
                    }
                }
            }
            if (compute) {
                if (stride == 3) {
                    targets[0] += prod[0];
                    targets[numOutputs] += prod[1];
                } else {
                    targets[0] = prod[0];
                    targets[numOutputs] = prod[1];
                }
            }
//            return;

            targets += !checkOutputBounds || x < numOutputsX - 16 ? 16 : numOutputsX - x;
        }

        targets += MUL24(3, numOutputsX);
    }
}


/*
 * This one uses a dynamic block size. The idea is that the block size should divide
 * the number of outputs, so that no iteration is wasted.
 */
template<int filterSize, int stride>
__global__ void conv_bw_fit_dyn_2per(float* imgs, float* filters, float* targets, int imgSize) {
    const int shImgSizeX = filterSize + blockDim.x - 1, shImgSizeY = filterSize + blockDim.y - 1;
    const int shImgPixels = MUL24(shImgSizeX, shImgSizeY); // size of shared buffer for image
    extern __shared__ float mem[];
    float *shImg = mem;
    float *shFilter = mem + shImgPixels;

    const int imgIdx = blockIdx.x;
    const int filtIdx = 2 * MUL24(blockDim.z, blockIdx.y) + 2 * threadIdx.z;
    const int numFilters = 2 * MUL24(blockDim.z, gridDim.y); // TODO: not exactly
    const int pidx = MUL24(threadIdx.y, blockDim.x) + threadIdx.x; // thread's index within the 4x16 "plate" of threads
    const int tidx = MUL24(threadIdx.z, MUL24(blockDim.y, blockDim.x)) + pidx; // thread's index within its block
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = MUL24(numOutputsX, numOutputsX);
    const int imgPixels = MUL24(imgSize, imgSize); // size of image

    const int filterPixels = filterSize * filterSize;
    const int loadX = tidx % (shImgSizeX);
    const int loadY = tidx / (shImgSizeX);

    imgs += MUL24(imgIdx, MUL24(stride, imgPixels)) + MUL24(loadY, imgSize) + loadX;
    filters += MUL24(filtIdx, MUL24(stride, filterPixels)) + pidx;
    targets += MUL24(imgIdx, MUL24(numFilters, numOutputs)) + MUL24(filtIdx, numOutputs) + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;

    float* shFilterLoad = &shFilter[MUL24(threadIdx.z, 2 * filterPixels) + pidx];
    float* shFilterLoad2 = &shFilter[MUL24((threadIdx.z * 2 + 1), filterPixels) + pidx];
    float* shImgLoad = &shImg[MUL24(loadY, shImgSizeX) + loadX];

    for (int i = pidx; i < filterPixels; i += MUL24(blockDim.x, blockDim.y)) { // Load the filter
        shFilterLoad[0] = filters[0];
        shFilterLoad2[0] = filters[MUL24(filterPixels, stride)];
        filters += MUL24(blockDim.x, blockDim.y);
        shFilterLoad += MUL24(blockDim.x, blockDim.y);
        shFilterLoad2 += MUL24(blockDim.x, blockDim.y);
    }

    for(int y = 0; y < numOutputsX; y += blockDim.y) {
        for(int x = 0; x < numOutputsX; x += blockDim.x) {
            __syncthreads();
            /*
             * This will load the entire (shared) image AS LONG AS THE FILTER SIZE IS <= 14.
             * If the filter size gets too big there won't be enough threads in the block
             * to load the entire shared image memory in one go. But that's ok because
             * when the filter size is > 14, this function doesn't have enough
             * memory to run anyway.
             */
            if (tidx < shImgPixels)
                shImgLoad[0] = imgs[x];

            __syncthreads();

            if (y + threadIdx.y < numOutputsX) {
                float* myShFilter = &shFilter[2 * MUL24(threadIdx.z, filterPixels)];
                float* myShImg = &shImg[MUL24(threadIdx.y, shImgSizeX) + threadIdx.x];
                float prod[2] = { 0, 0 };
//                targets[0] = shFilter[threadIdx.x];
//                return;
                for (int i = 0; i < filterSize; i++) {
                    for (int j = 0; j < filterSize; j++) {
                        prod[0] += myShFilter[0] * myShImg[0];
                        prod[1] += myShFilter[filterPixels] * myShImg[0];

                        myShFilter++;
                        myShImg++;
                    }
                    myShImg += blockDim.x - 1;
                }

                if (stride == 1) {
                    targets[0] = prod[0];
                    targets[numOutputs] = prod[1];
                } else {
                    targets[0] += prod[0];
                    targets[numOutputs] += prod[1];
                }
//                return;
            }
            targets += blockDim.x;
        }
        targets += MUL24(blockDim.y - 1, numOutputsX);
        imgs += imgSize * blockDim.y;
    }
}


/*
 * This one uses a dynamic block size. The idea is that the block size should divide
 * the number of outputs, so that no iteration is wasted.
 *
 * Uses 4*2*bZ*bX*bY^2 + 4*(2*bX - 1)*(2*bY - 1) bytes of shared memory
 * Assumes that the number of outputs in the x direction is divisible by blockDim.x,
 * but no assumption is made about blockDim.y.
 *
 * TODO: bugged, use 1per for template
 * TODO: uses too many registers to run :(
 */
template<bool checkFilterBounds, int bY, int bX, int stride>
__global__ void conv_bw_nofit_dyn_2per(float* imgs, float* filters, float* targets,
                                       const int imgSize, const int filterSize, const int numFilters) {
    const int shImgSizeX = 2*bX - 1, shImgSizeY = 2*bY - 1;
    const int shImgPixels = MUL24(shImgSizeX, shImgSizeY); // size of shared buffer for image
    extern __shared__ float mem[];
    float *shImg = mem;
    float *shFilter = mem + shImgPixels;

    const int imgIdx = blockIdx.x;
    const int filtIdx = 2 * MUL24(blockDim.z, blockIdx.y) + 2 * threadIdx.z;
//    const int numFilters = 2 * MUL24(blockDim.z, gridDim.y); // TODO: not exactly
    const int pidx = MUL24(threadIdx.y, bX) + threadIdx.x; // thread's index within the 4x16 "plate" of threads
    const int tidx = MUL24(threadIdx.z, bY*bX) + pidx; // thread's index within its block
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = MUL24(numOutputsX, numOutputsX);
    const int imgPixels = MUL24(imgSize, imgSize); // size of image

    const int filterPixels = MUL24(filterSize, filterSize);
    const int loadX = tidx % (shImgSizeX);
    const int loadY = tidx / (shImgSizeX);
    imgs += MUL24(imgIdx, MUL24(stride, imgPixels)) + loadY * imgSize + loadX;
    filters += MUL24(filtIdx, MUL24(stride, filterPixels)) + MUL24(threadIdx.y, filterSize) + threadIdx.x;
    targets += MUL24(imgIdx, MUL24(numFilters, numOutputs)) + MUL24(filtIdx, numOutputs) + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;

    float* shFilterLoad = &shFilter[MUL24(threadIdx.z, 2 * bX*bY) + pidx];
    float* shFilterLoad2 = &shFilter[MUL24((threadIdx.z * 2 + 1), bX*bY) + pidx];

    for(int y = 0; y < numOutputsX; y += bY) {
        for(int x = 0; x < numOutputsX; x += bX) {
            float prod[2] = { 0, 0 };
            const bool compute = filtIdx < numFilters;
            for(int fY = 0; fY < filterSize; fY += bY) {
                for(int fX = 0; fX < filterSize; fX += bX) {
                    __syncthreads();
                    if (compute) {
                        shFilterLoad[0] = 0;
                        shFilterLoad2[0] = 0;
                        if (!checkFilterBounds || (threadIdx.x + fX < filterSize && threadIdx.y + fY < filterSize)) {
                            shFilterLoad[0] = filters[MUL24(fY, filterSize) + fX];
                            shFilterLoad2[0] = filters[filterPixels*stride + MUL24(fY, filterSize) + fX];
                        }
                    }

                    for (int i = tidx; i < shImgPixels; i += bX*bY*blockDim.z) {
//                    if(tidx < shImgPixels)
                        if(i / shImgPixels + y + fY < imgSize)
                            shImg[i] = imgs[(y+fY) * imgSize + x + fX]; // TODO: this is wrong, does not change with i
                    }

                   __syncthreads();


                   if (compute && y + threadIdx.y < numOutputsX) {
                       float* myShFilter = &shFilter[2 * MUL24(threadIdx.z, bX*bY)];
//                       targets[0] = shImg[threadIdx.x];
//                       return;
                       float* myShImg = &shImg[MUL24(threadIdx.y, shImgSizeX) + threadIdx.x];

                       // Changing these to shorts reduced register usage by 2
#pragma unroll
                       for (short i = 0; i < bY; i++) {
                           for (short j = 0; j < bX; j++) {
                               prod[0] += myShFilter[0] * myShImg[0];
                               prod[1] += myShFilter[bX*bY] * myShImg[0];

                               myShFilter++;
                               myShImg++;
                           }
                           myShImg += bX - 1;
                       }
                   }
                }
            }
            if (compute) {
                if (stride == 1) {
                    targets[0] = prod[0];
                    targets[numOutputs] = prod[1];
                } else {
                    targets[0] += prod[0];
                    targets[numOutputs] += prod[1];
                }
            }
            targets += bX;
        }
        targets += MUL24(bY - 1, numOutputsX);
//        imgs += imgSize * bY;
    }
}

/*
 * This one uses a dynamic block size. The idea is that the block size should divide
 * the number of outputs, so that no iteration is wasted.
 *
 * Uses 4*bZ*bX*bY^2 + 4*(2*bX - 1)*(2*bY - 1) bytes of shared memory
 * Assumes that the number of outputs in the x direction is divisible by blockDim.x,
 * but no assumption is made about blockDim.y.
 */
template<bool checkFilterBounds, int bY, int bX, int stride>
__global__ void conv_bw_nofit_dyn_1per(float* imgs, float* filters, float* targets,
                                       const int imgSize, const int filterSize, const int numFilters) {
    const int shImgSizeX = 2*bX - 1, shImgSizeY = 2*bY - 1;
    const int shImgPixels = MUL24(shImgSizeX, shImgSizeY); // size of shared buffer for image
    extern __shared__ float mem[];
    float *shImg = mem;
    float *shFilter = mem + shImgPixels;

    const int imgIdx = blockIdx.x;
    const int filtIdx =  MUL24(blockDim.z, blockIdx.y) + threadIdx.z;
//    const int numFilters = 2 * MUL24(blockDim.z, gridDim.y); // TODO: not exactly
    const int pidx = MUL24(threadIdx.y, bX) + threadIdx.x; // thread's index within the 4x16 "plate" of threads
    const int tidx = MUL24(threadIdx.z, MUL24(bY, bX)) + pidx; // thread's index within its block
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = MUL24(numOutputsX, numOutputsX);
    const int imgPixels = MUL24(imgSize, imgSize); // size of image

    const int filterPixels = filterSize * filterSize;
    const int shFilterPixels = bY * bX;
    const int loadX = tidx % (shImgSizeX);
    const int loadY = tidx / (shImgSizeX);
    imgs += MUL24(imgIdx, MUL24(stride, imgPixels)) + loadY * imgSize + loadX;
    filters += MUL24(filtIdx, MUL24(stride, filterPixels)) + MUL24(threadIdx.y, filterSize) + threadIdx.x;
    targets += MUL24(imgIdx, MUL24(numFilters, numOutputs)) + MUL24(filtIdx, numOutputs) + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;

    float* shFilterLoad = &shFilter[MUL24(threadIdx.z, shFilterPixels) + pidx];
//    float* shImgLoad = &shImg[MUL24(loadY, shImgSizeX) + loadX];
    for(int y = 0; y < numOutputsX; y += bY) {
        for(int x = 0; x < numOutputsX; x += bX) {
            float prod = 0;
            const bool compute = filtIdx < numFilters;
            for(int fY = 0; fY < filterSize; fY += bY) {
                for(int fX = 0; fX < filterSize; fX += bX) {
                    __syncthreads();
                    if (compute) {
                        shFilterLoad[0] = 0;
                        if (!checkFilterBounds || (threadIdx.x + fX < filterSize && threadIdx.y + fY < filterSize)) {
                            shFilterLoad[0] = filters[MUL24(fY, filterSize) + fX];
                        }
                    }

                    for (int i = tidx; i < shImgPixels; i += bX*bY*blockDim.z) {
//                    if(tidx < shImgPixels)
                        if(i / shImgPixels + y + fY < imgSize)
                            shImg[i] = imgs[(y+fY) * imgSize + x + fX];
                    }

                   __syncthreads();


                   if (compute && y + threadIdx.y < numOutputsX) {
                       float* myShFilter = &shFilter[MUL24(threadIdx.z, shFilterPixels)];
//                       targets[0] = shFilter[threadIdx.x];
//                       return;
                       float* myShImg = &shImg[MUL24(threadIdx.y, shImgSizeX) + threadIdx.x];

                       // Changing these to shorts reduced register usage by 2
#pragma unroll
                       for (short i = 0; i < bY; i++) {
                           for (short j = 0; j < bX; j++) {
                               prod += myShFilter[0] * myShImg[0];

                               myShFilter++;
                               myShImg++;
                           }
                           myShImg += bX - 1;
                       }
                   }
                }
            }
            if (compute) {
                if (stride == 1) {
                    targets[0] = prod;
                } else {
                    targets[0] += prod;
                }
            }
            targets += bX;
        }
        targets += MUL24(bY - 1, numOutputsX);
//        imgs += imgSize * bY;
    }
}


#endif /* CONV_EXTRAS_CUH_ */
