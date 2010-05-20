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
 * nvmatrix_kernel.cu
 *
 *  Created on: 21-Jan-2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include "nvmatrix_kernel.cuh"

__global__ void kExp(float* gData, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = __expf(gData[i]);
}

__global__ void kLogistic1(float* gData, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = (1 + tanhf(gData[i] / 2)) / 2;
}

__global__ void kLogistic2(float* gData, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = 1 / (1 + expf(-gData[i]));
}

__global__ void kLog(float* gData, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = __logf(gData[i]);
}

__global__ void kSquare(float* gData, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = gData[i] * gData[i];
}

__global__ void kSqrt(float* gData, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = sqrtf(gData[i]);
}

__global__ void kZero(float* gData, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = 0;
}

__global__ void kReciprocal(float* gData, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = 1 / gData[i];
}

__global__ void kSign(float* gData, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = (gData[i] > 0) - (gData[i] < 0);
}

__global__ void kSubtractFromScalar(float* gData, float scalar, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = scalar - gData[i];
}

__global__ void kAddScalar(float* gData, float scalar, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = scalar + gData[i];
}

__global__ void kBiggerThanScalar(float* gData, float scalar, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = gData[i] > scalar;
}

__global__ void kSmallerThanScalar(float* gData, float scalar, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = gData[i] < scalar;
}

__global__ void kInRangeInc(float* gData, float lower, float upper, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = gData[i] >= lower && gData[i] <= upper;
}

__global__ void kInRangeExc(float* gData, float lower, float upper, float* target, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
        target[i] = gData[i] > lower && gData[i] < upper;
}

__global__ void kRandomUniform(unsigned int* rndMults, unsigned long long* rndWords, float* gData, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    for (unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        gData[i] = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
    }
    rndWords[idx] = rndWord;
}

__global__ void kBinarizeProbs(unsigned int* rndMults, unsigned long long* rndWords, float *gData, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    for (unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        gData[i] = gData[i] > (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
    }
    rndWords[idx] = rndWord;
}

#define PI 3.1415926535897932f

/*
 * TODO: modify to take mean/stdev
 */
__global__ void kAddGaussianNoise(unsigned int* rndMults, unsigned long long* rndWords, float* gData, const float stdev,
        unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    float rnd1, rnd2, R, T;
    for (unsigned int i = idx; i < numElements; i += 2 * NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd1 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd2 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        T = 2 * PI * rnd2;
        R = sqrtf(-2 * __logf(rnd1));
        gData[i] += stdev * R * __cosf(T);
        if (i + NUM_RND_STREAMS < numElements)
            gData[i + NUM_RND_STREAMS] += stdev * R * __sinf(T);
    }
    rndWords[idx] = rndWord;
}

__global__ void kAddGaussianNoise(unsigned int* rndMults, unsigned long long* rndWords, float* gData, const float* stdevs,
        unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    float rnd1, rnd2, R, T;
    for (unsigned int i = idx; i < numElements; i += 2 * NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd1 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd2 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        T = 2 * PI * rnd2;
        R = sqrtf(-2 * __logf(rnd1));
        gData[i] += stdevs[i] * R * __cosf(T);
        if (i + NUM_RND_STREAMS < numElements)
            gData[i + NUM_RND_STREAMS] += stdevs[i + NUM_RND_STREAMS] * R * __sinf(T);
    }
    rndWords[idx] = rndWord;
}

/*
 * TODO: modify to take mean/stdev
 */
__global__ void kRandomGaussian(unsigned int* rndMults, unsigned long long* rndWords, float* gData, const float stdev,
        unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    float rnd1, rnd2, R, T;
    for (unsigned int i = idx; i < numElements; i += 2 * NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd1 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd2 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        T = 2 * PI * rnd2;
        R = sqrtf(-2 * __logf(rnd1));
        gData[i] = stdev * R * __cosf(T);
        if (i + NUM_RND_STREAMS < numElements)
            gData[i + NUM_RND_STREAMS] = stdev * R * __sinf(T);
    }
    rndWords[idx] = rndWord;
}

__global__ void kRandomGaussian(unsigned int* rndMults, unsigned long long* rndWords, float* gData, const float* stdevs,
        unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    float rnd1, rnd2, R, T;
    for (unsigned int i = idx; i < numElements; i += 2 * NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd1 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd2 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        T = 2 * PI * rnd2;
        R = sqrtf(-2 * __logf(rnd1));
        gData[i] = stdevs[i] * R * __cosf(T);
        if (i + NUM_RND_STREAMS < numElements)
            gData[i + NUM_RND_STREAMS] = stdevs[i + NUM_RND_STREAMS] * R * __sinf(T);
    }
    rndWords[idx] = rndWord;
}

__global__ void kSeedRandom(unsigned int* rndMults, unsigned long long* rndWords, unsigned int seed) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // The initial x is the seed and the initial carry is 1
    unsigned long long rndWord = ((unsigned long long) seed << 32) + 1;
    const unsigned int rndMult = rndMults[idx];
    /*
     * Run the chain for a few steps so that all the streams have a chance
     * to differentiate. They start out generating similar random numbers
     * because all the multipliers are similar.
     */
    for (unsigned int i = 0; i < NUM_RND_BURNIN; i++) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
    }
    rndWords[idx] = rndWord;
}

__global__ void kBiggerThan(float* gMat1, float* gMat2, float* gMatTarget, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
        gMatTarget[idx] = gMat1[idx] > gMat2[idx];
}

__global__ void kCopy(float* srcStart, float* destStart, const int copyWidth,
                      const int srcJumpWidth, const int destJumpWidth, const int numElements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < numElements; i += blockDim.x * gridDim.x) {
        destStart[(i / copyWidth) * destJumpWidth + i % copyWidth] = srcStart[(i / copyWidth) * srcJumpWidth + i % copyWidth];
    }
}

__device__ inline int getTransArrayIndex(unsigned int width, unsigned int height, unsigned int i) {
    return height * (i % width) + i / width;
}

/*
 * like above but assumes destination is transposed.
 * note that this is not efficient because there will be
 * memory transactions that are not coalesced.
 */
__global__ void kCopyToTransDestSlow(float* srcStart, float* destStart, unsigned int srcCopyWidth, unsigned int srcJumpWidth,
        unsigned int destJumpHeight, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
        destStart[getTransArrayIndex(srcCopyWidth, destJumpHeight, idx)] = srcStart[(idx / srcCopyWidth) * srcJumpWidth + idx
                % srcCopyWidth];
}

/*
 * a not transposed, b transposed.
 * coalesced reads and writes, no bank conflicts cause of the +1.
 */
__global__ void kCopyToTransDestFast(float* srcStart, float* destStart, unsigned int srcCopyWidth, unsigned int srcCopyHeight,
        unsigned int srcJumpSize, unsigned int destJumpSize) {
    //    const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
    //    const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;

    //    if(idxX < srcCopyWidth && idxY < srcCopyHeight) {
    const unsigned int srcReadIdx = (blockIdx.y * blockDim.y + threadIdx.y) * srcJumpSize + blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int destWriteIdx = (blockIdx.x * blockDim.x + threadIdx.y) * destJumpSize + blockIdx.y * blockDim.y + threadIdx.x;
    __shared__
    float smem[COPY_BLOCK_SIZE][COPY_BLOCK_SIZE + 1];

    smem[threadIdx.x][threadIdx.y] = srcStart[srcReadIdx];
    __syncthreads();

    destStart[destWriteIdx] = smem[threadIdx.y][threadIdx.x];
    //    }
}

__global__ void kAdd(float* a, float* b, float* dest, unsigned int numEls, float scaleA, float scaleB) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    //    const unsigned int idx = blockIdx.y * height + blockIdx.x * blockDim.x  + threadIdx.y*blockDim.x + threadIdx.x;
    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = scaleA * a[i] + scaleB * b[i];
    }
}

__global__ void kMult(float* a, float* b, float* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    //    const unsigned int idx = blockIdx.y * height + blockIdx.x * blockDim.x  + threadIdx.y*blockDim.x + threadIdx.x;
    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] * b[i];
    }
}

__global__ void kDivide(float* a, float* b, float* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    //    const unsigned int idx = blockIdx.y * height + blockIdx.x * blockDim.x  + threadIdx.y*blockDim.x + threadIdx.x;
    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = __fdividef(a[i], b[i]);
    }
}

__global__ void kTranspose(float* a, float* dest, int width, int height) {
    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;
    const int tx = bx + threadIdx.x;
    const int ty = by + threadIdx.y;
    //    unsigned int idx = ty * width + tx;

    __shared__
    float smem[ADD_BLOCK_SIZE][ADD_BLOCK_SIZE + 1];

    if (tx < width && ty < height) {
        smem[threadIdx.y][threadIdx.x] = a[ty * width + tx];
    }
    __syncthreads();

    if (by + threadIdx.x < height && threadIdx.y + bx < width) {
        //        idx = height * (blockIdx.x * blockDim.x + threadIdx.y) + blockIdx.y * blockDim.y + threadIdx.x;
        dest[(bx + threadIdx.y) * height + by + threadIdx.x] = smem[threadIdx.x][threadIdx.y];
    }
}

__global__ void kSquaredDiff(float* a, float* b, float* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = (a[i] - b[i]) * (a[i] - b[i]);
    }
}

__global__ void kAdd3(float* a, const float* b, const float* c, const unsigned int numEls, const float scaleA, const float scaleB,
        const float scaleC) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numEls; i += numThreads) {
        a[i] = scaleA * a[i] + scaleB * b[i] + scaleC * c[i];
    }
}

__global__ void kTile(const float* src, float* tgt, const int srcWidth, const int srcHeight, const int tgtWidth, const int tgtHeight) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    //    const unsigned int numEls = tgtWidth * tgtHeight;
    for (unsigned int i = idx; i < tgtWidth * tgtHeight; i += numThreads) {
        const int y = i / tgtWidth;
        const int x = i % tgtWidth;
        const int srcY = y % srcHeight;
        const int srcX = x % srcWidth;
        tgt[i] = src[srcY * srcWidth + srcX];
    }
}

/*
 * Matrix in ROW-MAJOR order!
 */
__global__ void kAddRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height, float scaleVec) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] + scaleVec * vec[i % width];
    }
}

/*
 * Matrix in ROW-MAJOR order!
 */
__global__ void kAddColVector(float* mat, float* vec, float* tgtMat, const unsigned int width, const unsigned int height,
        const float scaleVec) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] + scaleVec * vec[i / width];
    }
}

/*
 * Matrix in ROW-MAJOR order!
 */
__global__ void kEqualsRowVector(float* mat, float* vec, float* tgtMat, const int width, const int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] == vec[i % width];
    }
}

/*
 * Matrix in ROW-MAJOR order!
 */
__global__ void kEqualsColVector(float* mat, float* vec, float* tgtMat, const int width, const int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] == vec[i / width];
    }
}

/*
 * Matrix in ROW-MAJOR order!
 */
__global__ void kBiggerThanRowVector(float* mat, float* vec, float* tgtMat, const int width, const int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] > vec[i % width];
    }
}

/*
 * Matrix in ROW-MAJOR order!
 */
__global__ void kBiggerThanColVector(float* mat, float* vec, float* tgtMat, const int width, const int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] > vec[i / width];
    }
}

/*
 * Matrix in ROW-MAJOR order!
 */
__global__ void kMultByRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] * vec[i % width];
    }
}

/*
 * Matrix in ROW-MAJOR order!
 */
__global__ void kMultByColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] * vec[i / width];
    }
}

/*
 * Matrix in ROW-MAJOR order!
 */
__global__ void kDivideByRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = __fdividef(mat[i], vec[i % width]);
    }
}

/*
 * Matrix in ROW-MAJOR order!
 */
__global__ void kDivideByColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = __fdividef(mat[i], vec[i / width]);
    }
}

/*
 * Bad when there are few columns. But if there are a few thousand columns, you can't really
 * go any faster than this because all the reads are coalesced and processor utilization is maximal.
 */
__global__ void kDumbSumCols(float* mat, float* vec, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    mat += idx;
    if (idx < width) {
        float sum = 0;
        for (int j = 0; j < height; j++) {
            sum += *mat;
            mat += width;
        }
        vec[idx] = sum;
    }
}

__global__ void kDumbMaxCols(float* mat, float* vec, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    mat += idx;
    if (idx < width) {
        float mx = *mat;
        mat += width;
        for (int j = 1; j < height; j++) {
            mx = myMax(*mat, mx);
            mat += width;
        }
        vec[idx] = mx;
    }
}

