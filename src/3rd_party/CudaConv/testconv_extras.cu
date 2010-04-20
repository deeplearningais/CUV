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
 * testconv_extras.cu
 *
 *  Created on: Nov 10, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#include <assert.h>
#include "testconv_extras.cuh"
#include "convCPU.h"

void test_conv_bw_fit_dyn_2per(int boardNum) {
    cudaSetDevice(boardNum > -1 ? boardNum : cutGetMaxGflopsDeviceId());
    cublasInit();
    NVMatrix::initDeviceProps();
    NVMatrix::initRandom(7);
    uint timer = 0;
    cutilCheckError( cutCreateTimer( &timer));

    int imgSize = 32, filterSize = 9;
    int numFilters = 64, numCases = 128;
    int filterPixels = filterSize * filterSize;
    int imgPixels = imgSize * imgSize;
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    assert(numFilters % 16 == 0);
    printf("Images: %d, filters: %d\n", numCases, numFilters);
    printf("Image size: %dx%d, filter size: %dx%d\n", imgSize, imgSize, filterSize, filterSize);
    printf("Color: no\n");

    Matrix filters(numFilters, filterPixels);
    Matrix images(numCases, imgPixels);
    Matrix targets(numCases, numFilters * numOutputs);
    filters.randomizeUniform();
    images.randomizeUniform();
    targets.apply(Matrix::ZERO);
    images.addScalar(1);

    NVMatrix nvFilters(filters, true);
    NVMatrix nvImages(images, true);
    NVMatrix nvTargets(targets, true); // eh why not

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    convCPU(images.getData(), filters.getData(), targets.getData(), imgSize, filterSize, numCases, numFilters);
    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 6);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    filters.print(3,3);
    dim3 threads(8,8,8);
    dim3 blocks(numCases, numFilters / 16);
    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    int shmem = 4*((filterSize + threads.x - 1)*(filterSize + threads.y - 1) + 2*threads.z * filterSize * filterSize);
    conv_bw_fit_dyn_2per<9, 1><<<blocks, threads, shmem>>>(nvImages.getDevData(), nvFilters.getDevData(), nvTargets.getDevData(), imgSize);
    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 6);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}


void test_conv_bw_nofit_dyn_1per(int imgSize, int filterSize, int threadsY, int threadsX, int boardNum) {
    cudaSetDevice(boardNum > -1 ? boardNum : cutGetMaxGflopsDeviceId());
    cublasInit();
    NVMatrix::initDeviceProps();
    NVMatrix::initRandom(7);
    uint timer = 0;
    cutilCheckError( cutCreateTimer( &timer));

//    int imgSize = 32, filterSize = 9;
    int numFilters = 64, numCases = 128;
    int filterPixels = filterSize * filterSize;
    int imgPixels = imgSize * imgSize;
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    assert(numFilters % 16 == 0);
    assert(numOutputsX % threadsX == 0);
    printf("Images: %d, filters: %d\n", numCases, numFilters);
    printf("Image size: %dx%d, filter size: %dx%d\n", imgSize, imgSize, filterSize, filterSize);
    printf("Color: no\n");

    Matrix filters(numFilters, filterPixels);
    Matrix images(numCases, imgPixels);
    Matrix targets(numCases, numFilters * numOutputs);
    filters.randomizeUniform();
    images.randomizeUniform();
    targets.apply(Matrix::ZERO);

    NVMatrix nvFilters(filters, true);
    NVMatrix nvImages(images, true);
    NVMatrix nvTargets(targets, true); // eh why not

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    convCPU(images.getData(), filters.getData(), targets.getData(), imgSize, filterSize, numCases, numFilters);
    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 6);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    filters.print(3,3);
    int threadsZ = int(512.0 / (threadsX * threadsY));
    int blocksY = int(ceil(float(numFilters) / (threadsZ)));
    bool checkFilterBounds = numOutputsX % filterSize != 0;
    assert(threadsZ > 0);
    dim3 threads(threadsX,threadsY,threadsZ);
    dim3 blocks(numCases, blocksY);
    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    int shmem = 4*((2*threads.x - 1)*(2*threads.y - 1) + threads.z * threads.x * threads.y);
    printf("Running %dx%d grid with %dx%dx%d blocks and %d bytes of shmem.\n", blocks.x, blocks.y, threads.x, threads.y, threads.z, shmem);
    if(checkFilterBounds) {
        conv_bw_nofit_dyn_1per<true,9,9, 1><<<blocks, threads, shmem>>>(nvImages.getDevData(), nvFilters.getDevData(), nvTargets.getDevData(),
                                                                    imgSize, filterSize, numFilters);
    } else {
        conv_bw_nofit_dyn_1per<false,9,9, 1><<<blocks, threads, shmem>>>(nvImages.getDevData(), nvFilters.getDevData(), nvTargets.getDevData(),
                                                                     imgSize, filterSize, numFilters);
    }

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 6);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}


void test_conv_bw_nofit_dyn_2per(int imgSize, int filterSize, int threadsY, int threadsX, int boardNum) {
    cudaSetDevice(boardNum > -1 ? boardNum : cutGetMaxGflopsDeviceId());
    cublasInit();
    NVMatrix::initDeviceProps();
    NVMatrix::initRandom(7);
    uint timer = 0;
    cutilCheckError( cutCreateTimer( &timer));

//    int imgSize = 32, filterSize = 9;
    int numFilters = 64, numCases = 128;
    int filterPixels = filterSize * filterSize;
    int imgPixels = imgSize * imgSize;
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    assert(numFilters % 16 == 0);
    assert(numOutputsX % threadsX == 0);
    printf("Images: %d, filters: %d\n", numCases, numFilters);
    printf("Image size: %dx%d, filter size: %dx%d\n", imgSize, imgSize, filterSize, filterSize);
    printf("Color: no\n");

    Matrix filters(numFilters, filterPixels);
    Matrix images(numCases, imgPixels);
    Matrix targets(numCases, numFilters * numOutputs);
    filters.randomizeUniform();
    images.randomizeUniform();
    targets.apply(Matrix::ZERO);

    NVMatrix nvFilters(filters, true);
    NVMatrix nvImages(images, true);
    NVMatrix nvTargets(targets, true); // eh why not

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    convCPU(images.getData(), filters.getData(), targets.getData(), imgSize, filterSize, numCases, numFilters);
    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, 0, 6);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));

    filters.print(3,3);
    int threadsZ = int(512.0 / (threadsX * threadsY));
    int blocksY = int(ceil(float(numFilters) / (2*threadsZ)));
    assert((numFilters % (threadsZ*2)) % 2 == 0);
    bool checkFilterBounds = numOutputsX % filterSize != 0;
    assert(threadsZ > 0);
    dim3 threads(threadsX,threadsY,threadsZ);
    dim3 blocks(numCases, blocksY);
    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    int shmem = 4*((2*threads.x - 1)*(2*threads.y - 1) + 2*threads.z * threads.x * threads.y);
    printf("Running %dx%d grid with %dx%dx%d blocks and %d bytes of shmem.\n", blocks.x, blocks.y, threads.x, threads.y, threads.z, shmem);
    assert(threadsX == 9 && threadsY == 9);
    if(checkFilterBounds) {
        conv_bw_nofit_dyn_2per<true,9,9, 1><<<blocks, threads, shmem>>>(nvImages.getDevData(), nvFilters.getDevData(), nvTargets.getDevData(),
                                                                    imgSize, filterSize, numFilters);
    } else {
        conv_bw_nofit_dyn_2per<false,9,9, 1><<<blocks, threads, shmem>>>(nvImages.getDevData(), nvFilters.getDevData(), nvTargets.getDevData(),
                                                                     imgSize, filterSize, numFilters);
    }

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, 0, 6);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}


void test_conv_bw_nofit_4x16_dynfilter_2per(int imgSize, int filterSize, int filterCacheY, int filterCacheX, int boardNum) {
    cudaSetDevice(boardNum > -1 ? boardNum : cutGetMaxGflopsDeviceId());
    cublasInit();
    NVMatrix::initDeviceProps();
    NVMatrix::initRandom(7);
    uint timer = 0;
    cutilCheckError( cutCreateTimer( &timer));

//    int imgSize = 32, filterSize = 9;
    int numFilters = 64, numCases = 128;
    int filterPixels = filterSize * filterSize;
    int imgPixels = imgSize * imgSize;
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    assert(numFilters % 16 == 0);
    assert(filterSize % filterCacheX == 0);
    assert(filterSize % filterCacheY == 0);
    printf("Images: %d, filters: %d\n", numCases, numFilters);
    printf("Image size: %dx%d, filter size: %dx%d\n", imgSize, imgSize, filterSize, filterSize);
    printf("Color: no\n");

    Matrix filters(numFilters, filterPixels);
    Matrix images(numCases, imgPixels);
    Matrix targets(numCases, numFilters * numOutputs);
    filters.randomizeUniform();
    images.randomizeUniform();
    targets.apply(Matrix::ZERO);

    NVMatrix nvFilters(filters, true);
    NVMatrix nvImages(images, true);
    NVMatrix nvTargets(targets, true); // eh why not

    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));
    convCPU(images.getData(), filters.getData(), targets.getData(), imgSize, filterSize, numCases, numFilters);
    cutilCheckError( cutStopTimer( timer));
    printf("CPU (partial) result:\n");
    targets.print(0, 3, numOutputsX-10, 6);
    printf("CPU time: %.6f msec\n", cutGetTimerValue(timer));


    bool checkOutputBounds = numOutputsX % 16 != 0;
    int blocksY = numFilters / 16;
    dim3 threads(16,4,8);
    dim3 blocks(numCases, blocksY);
    assert(filterCacheY == 8);
    assert(filterCacheX == 16);
    int shmem = (16 + filterCacheX - 1)*(4 + filterCacheY - 1) + 16*filterCacheX*filterCacheY;
    assert(shmem < 4096);
    printf("Filter cache size: %dx%d\n", filterCacheX, filterCacheY);
    printf("Using %d bytes of shared memory\n", shmem*4);
    cutilCheckError( cutResetTimer( timer));
    cutilCheckError( cutStartTimer( timer));

    if(checkOutputBounds) {
        conv_bw_nofit_4x16_dynfilter_2per<true,8,16, 1><<<blocks, threads>>>(nvImages.getDevData(), nvFilters.getDevData(), nvTargets.getDevData(),
                                                                    imgSize, filterSize);
    } else {
        conv_bw_nofit_4x16_dynfilter_2per<false,8,16, 1><<<blocks, threads>>>(nvImages.getDevData(), nvFilters.getDevData(), nvTargets.getDevData(),
                                                                     imgSize, filterSize);
    }

    cudaThreadSynchronize();
    cutilCheckError( cutStopTimer( timer));
    printf("GPU (partial) result:\n");
    nvTargets.print(0, 3, numOutputsX-10, 6);
    printf("GPU time: %.6f msec\n", cutGetTimerValue(timer));

    // Compare results
    Matrix cpuNVTargets(targets);
    nvTargets.copyToHost(cpuNVTargets);
    cpuNVTargets.subtract(targets);
    cpuNVTargets.apply(Matrix::ABS);
    printf("Max diff between CPU/GPU: %.6f\n", cpuNVTargets.max());
}
