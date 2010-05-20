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

void convolve(NVMatrix* images, NVMatrix* filters, NVMatrix* targets, int numGroups, bool color);
//void convolve_color(NVMatrix* images, NVMatrix* filters, NVMatrix* targets, int numGroups);

/*
 * This version uses block size (z, y, x) = dimZx4x16.
 * dimZ is one of 2, 4, 8.
 *
 * Each block convolves 1 image with 16 filters.
 * Works for filters 14x14 or smaller; image size only influences checkBounds.
 * The checked version uses 20 registers...would be nice to get that down to 16.
 *
 */
template<int filterSize, int stride, int dimZ, bool conv2>
__global__ void conv_bw_fit_4x16_2per(float* imgs, float* filters, float* targets,
                                      const int imgSize, const int numFiltersPerGroup, const int numGroups) {
    const int shImgSizeX = filterSize + 15, shImgSizeY = filterSize + 3;
    __shared__ float shImg[shImgSizeY][shImgSizeX];
    __shared__ float shFilter[2*dimZ][filterSize][filterSize];

    const int numImgsPerGroup = gridDim.x / numGroups;
    const int imgIdxInGroup = blockIdx.x % numImgsPerGroup;
//    const int numFiltersPerGroup = 2 * 8 * gridDim.y;
    const int groupIdx = blockIdx.x / numImgsPerGroup;
    const int filtIdxInGroup =  2 * dimZ * blockIdx.y + 2 * threadIdx.z;

    const int pidx = threadIdx.y * 16 + threadIdx.x; // thread's index within the 4x16 "plate" of threads
    const int tidx = threadIdx.z * 4 * 16 + pidx; // thread's index within its block
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = MUL24(numOutputsX, numOutputsX);
    const int imgPixels = MUL24(imgSize, imgSize); // size of image

    const int shImgPixels = shImgSizeX * shImgSizeY; // size of shared buffer for image
    const int filterPixels = filterSize * filterSize;

    if(!conv2) {
        imgs += (MUL24(groupIdx, numImgsPerGroup) + imgIdxInGroup) * stride * imgPixels;
        filters += MUL24(groupIdx, numFiltersPerGroup) * stride * filterPixels
                 + filtIdxInGroup * stride * filterPixels + pidx;
        targets += MUL24(MUL24(groupIdx, numFiltersPerGroup), numImgsPerGroup) * numOutputs
                 + MUL24(filtIdxInGroup, numImgsPerGroup) * numOutputs
                 + imgIdxInGroup * numOutputs
                 + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;
    } else {
        imgs += MUL24(groupIdx, numImgsPerGroup) * imgPixels
              + imgIdxInGroup * imgPixels;
        filters += MUL24(MUL24(groupIdx, numFiltersPerGroup), (numImgsPerGroup / stride)) * filterPixels
                 + (imgIdxInGroup / stride) * filterPixels
                 + MUL24(filtIdxInGroup, (numImgsPerGroup/stride)) * filterPixels
                 + pidx;
        targets += MUL24(groupIdx, numFiltersPerGroup) * numOutputs * stride
                 + MUL24(MUL24((imgIdxInGroup / stride), numGroups), numFiltersPerGroup) * numOutputs * stride
                 + filtIdxInGroup * numOutputs * stride
                 + (imgIdxInGroup % stride) * numOutputs
                 + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;
    }

    const int loadX = tidx % shImgSizeX;
    const int loadY = tidx / shImgSizeX;
    if (dimZ * 16 * 4 >= shImgPixels) {
        imgs += MUL24(loadY, imgSize) + loadX;
    }

    float* shFilterLoad = &shFilter[threadIdx.z * 2][0][pidx];
    float* shFilterLoad2 = &shFilter[threadIdx.z * 2 + 1][0][pidx];
    float* shImgLoad = &shImg[loadY][loadX];

    if (filtIdxInGroup < numFiltersPerGroup) {
        for (int i = pidx; i < filterPixels; i += 16 * 4) { // Load the filter
            shFilterLoad[0] = filters[0];
            if(!conv2) {
                shFilterLoad2[0] = filters[MUL24(filterPixels, stride)];
            } else {
                shFilterLoad2[0] = filters[MUL24((numImgsPerGroup/stride), filterPixels)];
            }
            filters += 16 * 4;
            shFilterLoad += 16 * 4;
            shFilterLoad2 += 16 * 4;
        }
    }
//    if(blockIdx.x != 0 || blockIdx.y != 0) return;
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
            if (dimZ * 16 * 4 >= shImgPixels) {
                if (tidx < shImgPixels) {
                    if (x + loadX < imgSize && y + loadY < imgSize) {
                        shImgLoad[0] = imgs[y * imgSize + x];
                    }
                }
            } else {
                for (int i = tidx; i < shImgPixels; i += 16 * 4 * dimZ) {
                    const int loadX = i % shImgSizeX;
                    const int loadY = i / shImgSizeX;
                    if (x + loadX < imgSize && y + loadY < imgSize) {
                        shImg[0][i] = imgs[(loadY + y) * imgSize + loadX + x];
                    }
                }
            }

            __syncthreads();

            // Retarded: combining these 2 ifs into 1 uses 2 more registers.
            if (filtIdxInGroup < numFiltersPerGroup) {
                if (x + threadIdx.x < numOutputsX && y + threadIdx.y < numOutputsX) {
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

                    if (!conv2) {
                        if (stride == 1) {
                            targets[MUL24(y, numOutputsX) + x] += prod[0];
                            targets[MUL24(y, numOutputsX) + x + MUL24(numOutputs, numImgsPerGroup)] += prod[1];
                        } else {
                            targets[MUL24(y, numOutputsX) + x] += prod[0];
                            targets[MUL24(y, numOutputsX) + x + MUL24(numOutputs, numImgsPerGroup)] += prod[1];
                        }
                    } else {
                        targets[MUL24(y, numOutputsX) + x] += prod[0];
                        targets[MUL24(y, numOutputsX) + x + numOutputs * stride] += prod[1];
                    }
                }
            }
        }
    }
}


/*
 * This version uses block size (z, y, x) = dimZx4x16.
 * dimZ is one of 2, 4, 8.
 *
 * Each block convolves 1 image with 8 filters.
 * Works for filters 20x20 or smaller; image size only influences checkBounds.
 */
template<int filterSize, int stride, int dimZ, bool conv2>
__global__ void conv_bw_fit_4x16_1per(float* imgs, float* filters, float* targets,
                                      const int imgSize, const int numFiltersPerGroup, const int numGroups) {
    const int shImgSizeX = filterSize + 15, shImgSizeY = filterSize + 3;
    const int shImgPixels = shImgSizeX * shImgSizeY; // size of shared buffer for image
    __shared__ float shImg[shImgSizeY][shImgSizeX];
    __shared__ float shFilter[dimZ][filterSize][filterSize];

    const int numImgsPerGroup = gridDim.x / numGroups;
    const int imgIdxInGroup = blockIdx.x % numImgsPerGroup;
//    const int numFiltersPerGroup = 2 * 8 * gridDim.y;
    const int groupIdx = blockIdx.x / numImgsPerGroup;
    const int filtIdxInGroup =  dimZ * blockIdx.y + threadIdx.z;

    const int pidx = threadIdx.y * 16 + threadIdx.x; // thread's index within the 4x16 "plate" of threads
    const int tidx = threadIdx.z * 4 * 16 + pidx; // thread's index within its block
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = MUL24(numOutputsX, numOutputsX);
    const int imgPixels = MUL24(imgSize, imgSize); // size of image
    const int filterPixels = filterSize * filterSize;

    if (!conv2) {
        imgs += (MUL24(groupIdx, numImgsPerGroup) + imgIdxInGroup) * stride * imgPixels;
        filters += MUL24(groupIdx, numFiltersPerGroup) * stride * filterPixels
                 + filtIdxInGroup * stride * filterPixels + pidx;
        targets += MUL24(MUL24(groupIdx, numFiltersPerGroup), numImgsPerGroup) * numOutputs
                 + MUL24(filtIdxInGroup, numImgsPerGroup) * numOutputs
                 + imgIdxInGroup * numOutputs
                 + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;
    } else {
        imgs += MUL24(groupIdx, numImgsPerGroup) * imgPixels
              + imgIdxInGroup * imgPixels;
        filters += MUL24(MUL24(groupIdx, numFiltersPerGroup), (numImgsPerGroup / stride)) * filterPixels
                 + (imgIdxInGroup / stride) * filterPixels
                 + MUL24(filtIdxInGroup, (numImgsPerGroup/stride)) * filterPixels
                 + pidx;
        targets += MUL24(groupIdx, numFiltersPerGroup) * numOutputs * stride
                 + MUL24(MUL24((imgIdxInGroup / stride), numGroups), numFiltersPerGroup) * numOutputs * stride
                 + filtIdxInGroup * numOutputs * stride
                 + (imgIdxInGroup % stride) * numOutputs
                 + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;
    }

    float* shFilterLoad = &shFilter[threadIdx.z][0][pidx];
    if (filtIdxInGroup < numFiltersPerGroup) {
        for (int i = pidx; i < filterPixels; i += 16 * 4) { // Load the filter
            shFilterLoad[0] = filters[0];
            filters += 16 * 4;
            shFilterLoad += 16 * 4;
        }
    }

    for(int y = 0; y < numOutputsX; y += 4) {
        for(int x = 0; x < numOutputsX; x += 16) {
            __syncthreads();
            for (int i = tidx; i < shImgPixels; i += 16 * 4 * dimZ) {
                const int loadX = i % (shImgSizeX);
                const int loadY = i / (shImgSizeX);
                if (x + loadX < imgSize && y + loadY < imgSize) {
                    shImg[0][i] = imgs[(loadY + y) * imgSize + loadX + x];
                }
            }

            __syncthreads();
            if (filtIdxInGroup < numFiltersPerGroup) {
                if (x + threadIdx.x < numOutputsX && y + threadIdx.y < numOutputsX) {
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
                    if (!conv2) {
                        if(stride == 1) {
                            targets[MUL24(y, numOutputsX) + x] += prod;
                        } else {
                            targets[MUL24(y, numOutputsX) + x] += prod;
                        }
                    } else {
                        targets[MUL24(y, numOutputsX) + x] += prod;
                    }
                }
            }
        }
    }
}

/*
 * This version uses block size (z, y, x) = 8x4x16.
 *
 * Each block convolves 1 image with 16 filters.
 * Use only when the filter size is > 14, otherwise use the functions that
 * cache the entire filter.
 */
template<bool checkFilterBounds, int stride, int dimZ, bool conv2>
__global__ void conv_bw_nofit_4x16_2per(float* imgs, float* filters, float* targets, const int imgSize, const int filterSize,
                                        const int numFiltersPerGroup, const int numGroups) {
    const int shImgSizeX = 16 + 15, shImgSizeY = 4 + 3;
    __shared__ float shImg[shImgSizeY][shImgSizeX];
    __shared__ float shFilter[2*dimZ][4][16];

    const int numImgsPerGroup = gridDim.x / numGroups;
    const int imgIdxInGroup = blockIdx.x % numImgsPerGroup;
    const int groupIdx = blockIdx.x / numImgsPerGroup;
    const int filtIdxInGroup =  2 * dimZ * blockIdx.y + 2 * threadIdx.z;

    const int pidx = threadIdx.y * 16 + threadIdx.x; // thread's index within the 4x16 "plate" of threads
    const int tidx = threadIdx.z * 4 * 16 + pidx; // thread's index within its block
    const int numOutputsX = imgSize - filterSize + 1;
    const int numOutputs = MUL24(numOutputsX, numOutputsX);
    const int imgPixels = MUL24(imgSize, imgSize); // size of image

    const int shImgPixels = shImgSizeY * shImgSizeX; // size of shared buffer for image
    const int filterPixels = MUL24(filterSize, filterSize);
    const int filterStride =  !conv2 ? MUL24(filterPixels, stride) : MUL24((numImgsPerGroup/stride), filterPixels);

    if(!conv2) {
        imgs += (MUL24(groupIdx, numImgsPerGroup) + imgIdxInGroup) * stride * imgPixels;
        filters += MUL24(groupIdx, numFiltersPerGroup) * stride * filterPixels
                 + filtIdxInGroup * stride * filterPixels
                 + MUL24(threadIdx.y, filterSize) + threadIdx.x;
        targets += MUL24(MUL24(groupIdx, numFiltersPerGroup), numImgsPerGroup) * numOutputs
                 + MUL24(filtIdxInGroup, numImgsPerGroup) * numOutputs
                 + imgIdxInGroup * numOutputs
                 + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;
    } else {
        imgs += MUL24(groupIdx, numImgsPerGroup) * imgPixels
              + imgIdxInGroup * imgPixels;
        filters += MUL24(MUL24(groupIdx, numFiltersPerGroup), (numImgsPerGroup / stride)) * filterPixels
                 + (imgIdxInGroup / stride) * filterPixels
                 + MUL24(filtIdxInGroup, (numImgsPerGroup/stride)) * filterPixels
                 + MUL24(threadIdx.y, filterSize) + threadIdx.x;
        targets += MUL24(groupIdx, numFiltersPerGroup) * numOutputs * stride
                 + MUL24(MUL24((imgIdxInGroup / stride), numGroups), numFiltersPerGroup) * numOutputs * stride
                 + filtIdxInGroup * numOutputs * stride
                 + (imgIdxInGroup % stride) * numOutputs
                 + MUL24(threadIdx.y, numOutputsX) + threadIdx.x;
    }

    const int loadX = tidx % shImgSizeX;
    const int loadY = tidx / shImgSizeX;
    if (dimZ * 16 * 4 >= shImgPixels) {
        imgs += MUL24(loadY, imgSize) + loadX;
    }

    float* shFilterLoad = &shFilter[threadIdx.z * 2][0][pidx];
    float* shFilterLoad2 = &shFilter[threadIdx.z * 2 + 1][0][pidx];
    float* shImgLoad = &shImg[loadY][loadX];
//if(blockIdx.x != 0 || blockIdx.y != 0) return;
    for(int y = 0; y < numOutputsX; y += 4) {
        for (int x = 0; x < numOutputsX; x += 16) {
            float prod[2] = { 0, 0 };
            const bool compute = filtIdxInGroup < numFiltersPerGroup && (x + threadIdx.x < numOutputsX && y + threadIdx.y < numOutputsX);
            for (int fY = 0; fY < filterSize; fY += 4) {
                for (int fX = 0; fX < filterSize; fX += 16) {

                    __syncthreads();
                    /*
                     * Load filter
                     */
                    if(filtIdxInGroup < numFiltersPerGroup) {
                        shFilterLoad[0] = 0;
                        shFilterLoad2[0] = 0;
                        if (!checkFilterBounds || (threadIdx.x + fX < filterSize && threadIdx.y + fY < filterSize)) {
                            float* f = &filters[MUL24(fY, filterSize) + fX];
                            shFilterLoad[0] = f[0];
                            shFilterLoad2[0] = f[filterStride];
                        }
                    }

                    /*
                     * Load image
                     */
                    if (dimZ * 16 * 4 >= shImgPixels) {
                        if (tidx < shImgPixels && (x + fX + loadX < imgSize && y + fY + loadY < imgSize)) {
                            // I tried incrementing imgs instead of indexing it, but that
                            // uses more registers and doesn't speed things up much.
                            shImgLoad[0] = imgs[MUL24((y + fY), imgSize) + x + fX];
                        }
                    } else {
                        for (int i = tidx; i < shImgPixels; i += 16 * 4 * dimZ) {
                            const int loadX = i % (shImgSizeX);
                            const int loadY = i / (shImgSizeX);
                            if (x + fX + loadX < imgSize && y + fY + loadY < imgSize) {
                                shImg[0][i] = imgs[MUL24(loadY + y + fY, imgSize) + loadX + x + fX];
                            }
                        }
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
                if (!conv2) {
                    if (stride == 1) {
                        targets[MUL24(y, numOutputsX) + x] += prod[0];
                        targets[MUL24(y, numOutputsX) + x + MUL24(numOutputs, numImgsPerGroup)] += prod[1];
                    } else {
                        targets[MUL24(y, numOutputsX) + x] += prod[0];
                        targets[MUL24(y, numOutputsX) + x + MUL24(numOutputs, numImgsPerGroup)] += prod[1];
                    }
                } else {
                    targets[MUL24(y, numOutputsX) + x] += prod[0];
                    targets[MUL24(y, numOutputsX) + x + numOutputs * stride] += prod[1];
                }
            }
        }
    }
}


/*
 * This function is suitable for cases when the number of outputs is small
 * (i.e. when the filter size is nearly equal to the image size).
 * This version uses a dynamic block size. bX and bY should be set
 * to the number of outputs (bX and bY are always equal).
 * bZ should be set such that bZ*bX*bY <= 512, but it's important that each
 * block have at least (2 * bX - 1)*(2 * bY - 1) threads.
 * IMPORTANT: bZ MUST be even. <-- wait why?
 *
 * Each block convolves 1 image with bZ*2 filters.
 *
 * This one loads the filter piecewise, even if it's very small. But this is
 * more or less ok for this routine because it nonetheless loads each filter only once.
 * This is because it always has as many threads as outputs, so it doesn't need
 * to loop to produce all the outputs.
 *
 * NOTE: 4per version is slower.
 */
template<bool checkFilterBounds, int stride, int bXY, int bZ, bool conv2>
__global__ void conv_bw_nofit_dynXYZ_2per(float* imgs, float* filters, float* targets,
                                          const int imgSize, const int filterSize, const int numFiltersPerGroup, const int numGroups) {
    const int shImgSizeXY = 2 * bXY - 1;
    __shared__ float shImg[shImgSizeXY][shImgSizeXY];
    __shared__ float shFilter[2 * bZ][bXY][bXY];

    const int numImgsPerGroup = gridDim.x / numGroups;
    const int imgIdxInGroup = blockIdx.x % numImgsPerGroup;
    const int groupIdx = blockIdx.x / numImgsPerGroup;
    const int filtIdxInGroup =  2 * bZ * blockIdx.y + 2 * threadIdx.z;

    const int pidx = threadIdx.y * bXY + threadIdx.x; // thread's index within the bYxbX "plate" of threads
    const int tidx = threadIdx.z * (bXY * bXY)  + pidx; // thread's index within its block
    const int imgPixels = MUL24(imgSize, imgSize); // size of image

    const int shImgPixels = shImgSizeXY * shImgSizeXY; // size of shared buffer for image
    const int filterPixels = MUL24(filterSize, filterSize);
    const int loadX = tidx % shImgSizeXY;
    const int loadY = tidx / shImgSizeXY;
    const bool load = tidx < shImgPixels;
    const int cmpX = imgSize - loadX, cmpY = imgSize - loadY;
    const int filterStride = !conv2 ? MUL24(filterPixels, stride) : MUL24((numImgsPerGroup/stride), filterPixels);
    if (!conv2) {
        imgs += (MUL24(groupIdx, numImgsPerGroup) + imgIdxInGroup) * stride * imgPixels
                + MUL24(loadY, imgSize) + loadX;
        filters += MUL24(groupIdx, numFiltersPerGroup) * stride * filterPixels
                 + filtIdxInGroup * stride * filterPixels
                 + MUL24(threadIdx.y, filterSize) + threadIdx.x;
        targets += MUL24(MUL24(groupIdx, numFiltersPerGroup), numImgsPerGroup) * bXY * bXY
                 + MUL24(filtIdxInGroup, numImgsPerGroup) * bXY * bXY
                 + imgIdxInGroup * bXY * bXY
                 + MUL24(threadIdx.y, bXY) + threadIdx.x;
    } else {
        imgs += MUL24(groupIdx, numImgsPerGroup) * imgPixels
              + imgIdxInGroup * imgPixels
              + MUL24(loadY, imgSize) + loadX;
        filters += MUL24(MUL24(groupIdx, numFiltersPerGroup), (numImgsPerGroup / stride)) * filterPixels
                 + (imgIdxInGroup / stride) * filterPixels
                 + MUL24(filtIdxInGroup, (numImgsPerGroup/stride)) * filterPixels
                 + MUL24(threadIdx.y, filterSize) + threadIdx.x;
        targets += MUL24(groupIdx, numFiltersPerGroup) * bXY * bXY * stride
                 + MUL24(MUL24((imgIdxInGroup / stride), numGroups), numFiltersPerGroup) * bXY * bXY * stride
                 + filtIdxInGroup * bXY * bXY * stride
                 + (imgIdxInGroup % stride) * bXY * bXY
                 + MUL24(threadIdx.y, bXY) + threadIdx.x;

//        imgs += imgPixels * imgIdx + MUL24(loadY, imgSize) + loadX;
//        filters += MUL24((imgIdx / stride), numFilters) * filterPixels + filtIdx * filterPixels + MUL24(threadIdx.y, filterSize) + threadIdx.x;
//        targets += imgIdx * numFilters * (bXY * bXY) + filtIdx * (bXY * bXY) + MUL24(threadIdx.y, bXY) + threadIdx.x;
    }

    float* shFilterLoad = &shFilter[threadIdx.z * 2][0][pidx];
    float* shImgLoad = &shImg[loadY][loadX];
//    if(imgIdx > 383)
//        return;
    const bool compute = filtIdxInGroup < numFiltersPerGroup;
    float prod[2] = { 0, 0 };
    for (int fY = 0; fY < filterSize; fY += bXY) {
        for (int fX = 0; fX < filterSize; fX += bXY) {

            __syncthreads();
            if (compute) {
                shFilterLoad[0] = 0;
                shFilterLoad[bXY * bXY] = 0;
                if (!checkFilterBounds || (threadIdx.x + fX < filterSize && threadIdx.y + fY < filterSize)) {
                    const float* f = &filters[MUL24(fY, filterSize) + fX];
                    shFilterLoad[0] = f[0];
                    shFilterLoad[bXY * bXY] = f[filterStride]; // using filterSize here saves a register
                }
            }

            if (load && fX < cmpX && fY < cmpY) {
                shImgLoad[0] = imgs[MUL24(fY, imgSize) + fX];
            }

            __syncthreads();

            if (compute) {
                const float* myShFilter = &shFilter[2 * threadIdx.z][0][0];
                const float* myShImg = &shImg[threadIdx.y][threadIdx.x];
                #pragma unroll
                for (int i = 0; i < bXY; i++) {
                    #pragma unroll
                    for (int j = 0; j < bXY; j++) {
                        prod[0] += myShFilter[0] * myShImg[0];
                        prod[1] += myShFilter[bXY * bXY] * myShImg[0];

                        myShFilter++;
                        myShImg++;
                    }
                    myShImg += bXY - 1;
                }
            }
        }
//        imgs += MUL24(imgSize, bXY);
    }
    if (compute) {
        if(!conv2) {
            if (stride == 1) {
                targets[0] += prod[0];
                targets[(bXY * bXY) * numImgsPerGroup] += prod[1];
            } else {
                targets[0] += prod[0];
                targets[(bXY * bXY) * numImgsPerGroup] += prod[1];
            }
        } else {
            targets[0] += prod[0];
            targets[bXY * bXY * stride] += prod[1];
        }
    }
}

#endif /* CONV_CUH_ */
