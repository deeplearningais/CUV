/*
 * nvmatrix_kernel.h
 *
 *  Created on: 25-Jan-2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#ifndef NVMATRIX_KERNEL_H_
#define NVMATRIX_KERNEL_H_

#define NUM_BLOCKS_MAX                      65535

#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)

/*
 * Defines for getting the values at the lower and upper 32 bits
 * of a 64-bit number.
 */
#define LOW_BITS(x)                         ((x) & 0xffffffff)
#define HIGH_BITS(x)                        ((x) >> 32)

/*
 * Number of iterations to run random number generator upon initialization.
 */
#define NUM_RND_BURNIN                      1000

/*
 * Default grid/block sizes for the various functions.
 */
#define ADD_BLOCK_SIZE                      16
#define COPY_BLOCK_SIZE                     16

#define NUM_TILE_BLOCKS                     2048
#define NUM_TILE_THREADS_PER_BLOCK          512

#define NUM_APPLY_BLOCKS                    4096
#define NUM_APPLY_THREADS_PER_BLOCK         512

#define NUM_ADD_VECTOR_BLOCKS               4096
#define NUM_ADD_VECTOR_THREADS_PER_BLOCK    512

#define NUM_SUM_ROWS_THREADS_PER_BLOCK      512 /* THIS HAS TO BE A POWER OF 2! */
#define NUM_SUM_COLS_THREADS_PER_BLOCK      256

#define NUM_VECTOR_OP_BLOCKS                4096
#define NUM_VECTOR_OP_THREADS_PER_BLOCK     512

#define AGG_SHORT_ROWS_THREADS_X            16
#define AGG_SHORT_ROWS_THREADS_Y            32
#define AGG_SHORT_ROWS_LOOPS_Y              32
#define AGG_MAX                             0
#define AGG_SUM                             1

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

#ifndef MUL24
#define MUL24 __mul24
#endif

__global__ void kExp(float* gData, float* target, unsigned int numElements);
__global__ void kLogistic1(float* gData, float* target, unsigned int numElements);
__global__ void kLogistic2(float* gData, float* target, unsigned int numElements);
__global__ void kLog(float* gData, float* target, unsigned int numElements);
__global__ void kSquare(float* gData, float* target, unsigned int numElements);
__global__ void kSqrt(float* gData, float* target, unsigned int numElements);
__global__ void kZero(float* gData, float* target, unsigned int numElements);
__global__ void kReciprocal(float* gData, float* target, unsigned int numElements);
__global__ void kSubtractFromScalar(float* gData, float scalar, float* target, unsigned int numElements);
__global__ void kAddScalar(float* gData, float scalar, float* target, unsigned int numElements);
__global__ void kBiggerThanScalar(float* gData, float scalar, float* target, unsigned int numElements);
__global__ void kAddGaussianNoise(unsigned int* rndMults, unsigned long long* rndWords, float* gData, float stdev, unsigned int numElements);
__global__ void kRandomGaussian(unsigned int* rndMults, unsigned long long* rndWords, float* gData, float stdev, unsigned int numElements);
__global__ void kRandomUniform(unsigned int* randMults, unsigned long long* randWords, float* gData, unsigned int numElements);
__global__ void kBinarizeProbs(unsigned int* randMults, unsigned long long* randWords, float *gData, unsigned int numElements);
__global__ void kSeedRandom(unsigned int* randMults, unsigned long long* randWords, unsigned int seed);
__global__ void kBiggerThan(float* gMat1, float* gMat2, float* gMatTarget, unsigned int numElements);
__global__ void kCopy(float* srcStart, float* destStart, unsigned int copyWidth, unsigned int jumpWidth, unsigned int numElements);
__device__ inline int getTransArrayIndex(unsigned int width, unsigned int height, unsigned  int i);
__global__ void kCopyToTransDestSlow(float* srcStart, float* destStart, unsigned int srcCopyWidth,
                                    unsigned int srcJumpWidth, unsigned int destJumpHeight, unsigned int numElements);
__global__ void kCopyToTransDestFast(float* srcStart, float* destStart, unsigned int srcCopyWidth, unsigned int srcCopyHeight,
                                    unsigned int srcJumpSize, unsigned int destJumpSize);
__global__ void kAdd(float* a, float* b, float* dest,
                     unsigned int numEls, float scaleA, float scaleB);
__global__ void kAddTransSlow(float* a, float* b, float* dest, unsigned int width, unsigned int height,
                          unsigned int numEls, float scaleA, float scaleB);
__global__ void kAddTransFast(float* a, float* b, float* dest, unsigned int width, unsigned int height,
                           unsigned int bJumpWidth, float scaleA, float scaleB);
__global__ void kMultTransFast(float* a, float* b, float* dest, unsigned int width, unsigned int height, unsigned int bJumpWidth);
__global__ void kMult(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kDivideTransFast(float* a, float* b, float* dest, unsigned int width, unsigned int height, unsigned int bJumpWidth);
__global__ void kDivide(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kAdd3(float* a, const float* b, const float* c, const unsigned int numEls,
        const float scaleA, const float scaleB, const float scaleC);
__global__ void kSquaredDiffTransFast(float* a, float* b, float* dest, unsigned int width, unsigned int bJumpWidth);
__global__ void kSquaredDiff(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kTile(float* src, float* tgt, unsigned int srcWidth, unsigned int srcHeight, unsigned int tgtWidth,
                      unsigned int tgtHeight);
__global__ void kAddRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height, float scaleVec);
__global__ void kAddColVector(float* mat, float* vec, float* tgtMat, const unsigned int width, const unsigned int height, const float scaleVec);
__global__ void kMultByRowVector(float* mat, float* vec, float* tgtMat, unsigned int width,unsigned int height);
__global__ void kMultByColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void kDivideByRowVector(float* mat, float* vec, float* tgtMat, unsigned int width,unsigned int height);
__global__ void kDivideByColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void kDumbSumCols(float* mat, float* vec, unsigned int width, unsigned int height);
__global__ void kVeryDumbSumRows(float* mat, float* vec, unsigned int width, unsigned int height);
__global__ void kDumbMaxCols(float* mat, float* vec, unsigned int width, unsigned int height);
__global__ void kTranspose(float* a, float* dest, int width, int height);

/*
 * a := a + b + c where b, c might have different transposedness from a.
 */
template<bool checkBounds>
__global__ void kAddTrans3Fast(float* a, const float* b, const float* c, const unsigned int width, const unsigned int height,
                               const unsigned int transJumpWidth, const float scaleA, const float scaleB, const float scaleC,
                               const bool transB, const bool transC) {
    unsigned int idxYA = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idxXA = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int idxXB = blockIdx.x * blockDim.x + threadIdx.y;
    unsigned int idxYB = blockIdx.y * blockDim.y + threadIdx.x;

    __shared__ float smemB[ADD_BLOCK_SIZE][ADD_BLOCK_SIZE + 1];
    __shared__ float smemC[ADD_BLOCK_SIZE][ADD_BLOCK_SIZE + 1];

    if (!checkBounds || (idxYB < height && idxXB < width)) {
        const unsigned int bIdx = idxXB * height + idxYB;
        if (transB)
            smemB[threadIdx.x][threadIdx.y] = b[bIdx];

        if (transC)
            smemC[threadIdx.x][threadIdx.y] = c[bIdx];
    }

    __syncthreads();

    if(!checkBounds || (idxXA < width && idxYA < height)) {
        const unsigned int idx = idxYA * width + idxXA;
        a[idx] =    scaleA * a[idx] +
                    scaleB * (transB ? smemB[threadIdx.y][threadIdx.x] : b[idx]) +
                    scaleC * (transC ? smemC[threadIdx.y][threadIdx.x] : c[idx]);
    }
}

/*
 * This one gets coalesced reads but computes only a partial sum which
 * must either be summed again (recursively) or summed on the host.
 */
template<int blockSize>
__global__ void kSumRows(float* mat, float* matSum, int width, int height, int sumWidth) {
    const int idxX = blockIdx.x * blockSize*2 + threadIdx.x;

    __shared__ float accum[blockSize*2];

    matSum += blockIdx.y * sumWidth + blockIdx.x;
    /*
     * Here it's important to make sure that all threads in a block call __syncthreads,
     * so I have even the redundant threads (for which idxX >= width) enter this loop
     * just so that they may call __syncthreads at the appropriate times.
     */
    mat += width * blockIdx.y + idxX;
    float* myAccum = &accum[threadIdx.x];
    myAccum[0] = 0;
    myAccum[blockSize] = 0;
    for (int idxY = blockIdx.y; idxY < height; idxY += gridDim.y) {
        if (idxX < width) {
            myAccum[0] = mat[0];
            if(idxX + blockSize < width)
                myAccum[blockSize] = mat[blockSize];
        }
//
        if (blockSize >= 512) { // evaluated at compile-time
            __syncthreads();
            if (threadIdx.x < 512)
                myAccum[0] += myAccum[512];
        }
        if (blockSize >= 256) {
            __syncthreads();
            if (threadIdx.x < 256)
                myAccum[0] += myAccum[256];
        }
        if (blockSize >= 128) {
            __syncthreads();
            if (threadIdx.x < 128)
                myAccum[0] += myAccum[128];
        }
        if (blockSize >= 64) {
            __syncthreads();
            if (threadIdx.x < 64)
                myAccum[0] += myAccum[64];
        }

        __syncthreads();
        if (threadIdx.x < 32) { // executed only by first warp
            myAccum[0] += myAccum[32];
            myAccum[0] += myAccum[16];
            myAccum[0] += myAccum[8];
            myAccum[0] += myAccum[4];
            myAccum[0] += myAccum[2];
            myAccum[0] += myAccum[1];
        }

        if (threadIdx.x == 0) {
            matSum[0] = myAccum[0];
            matSum += gridDim.y * sumWidth;
        }
        __syncthreads();
        mat += width * gridDim.y;
    }
}

__device__ float myMax(float a, float b) {
    return a > b ? a : b;
}
/*
 * Note: looping over y dimension doesn't help.
 */
template<int blockSize>
__global__ void kMaxRows(float* mat, float* matSum, int width, int height, int sumWidth) {
    const int idxX = blockIdx.x * blockSize*2 + threadIdx.x;

    __shared__ float accum[blockSize*2];

    matSum += blockIdx.y * sumWidth + blockIdx.x;
    /*
     * Here it's important to make sure that all threads in a block call __syncthreads,
     * so I have even the redundant threads (for which idxX >= width) enter this loop
     * just so that they may call __syncthreads at the appropriate times.
     */
    mat += width * blockIdx.y + idxX;
    float* myAccum = &accum[threadIdx.x];
    myAccum[0] = -2e38;
    myAccum[blockSize] = -2e38;
    for (int idxY = blockIdx.y; idxY < height; idxY += gridDim.y) {
        if(idxX < width) {
            myAccum[0] = mat[0];
            if(idxX + blockSize < width)
                myAccum[blockSize] = mat[blockSize];
        }
//
        if (blockSize >= 512) { // evaluated at compile-time
            __syncthreads();
            if (threadIdx.x < 512)
                myAccum[0] = myMax(myAccum[0], myAccum[512]);
        }
        if (blockSize >= 256) {
            __syncthreads();
            if (threadIdx.x < 256)
                myAccum[0] = myMax(myAccum[0], myAccum[256]);
        }
        if (blockSize >= 128) {
            __syncthreads();
            if (threadIdx.x < 128)
                myAccum[0] = myMax(myAccum[0], myAccum[128]);
        }
        if (blockSize >= 64) {
            __syncthreads();
            if (threadIdx.x < 64)
                myAccum[0] = myMax(myAccum[0], myAccum[64]);
        }

        __syncthreads();
        if (threadIdx.x < 32) { // executed only by first warp
            myAccum[0] = myMax(myAccum[0], myAccum[32]);
            myAccum[0] = myMax(myAccum[0], myAccum[16]);
            myAccum[0] = myMax(myAccum[0], myAccum[8]);
            myAccum[0] = myMax(myAccum[0], myAccum[4]);
            myAccum[0] = myMax(myAccum[0], myAccum[2]);
            myAccum[0] = myMax(myAccum[0], myAccum[1]);
        }

        if (threadIdx.x == 0) {
            matSum[0] = myAccum[0];
            matSum += gridDim.y * sumWidth;
        }

        __syncthreads();
        mat += width * gridDim.y;
    }
}

/*
 * To be used when the rows are <= 64.
 * Block size (y, x) = (32, 16)
 *
 * TODO: try to reduce reg usage. i think this can be made faster too.
 */
//#define AGG_SHORT_ROWS_LOOPS_X  4
template <int AGG_TYPE, int LOOPS_X, int SUM_WIDTH_UPPERBOUND>
__global__ void kAggShortRows(float* mat, float* matSum, int width, int height) {
    const int shmemX = AGG_SHORT_ROWS_THREADS_X + 1;
    __shared__ float shmem[AGG_SHORT_ROWS_THREADS_Y*shmemX];

    const int tidx = threadIdx.y * AGG_SHORT_ROWS_THREADS_X + threadIdx.x;
    const int ty = LOOPS_X == 1 ? tidx / width : threadIdx.y;
    const int tx = LOOPS_X == 1 ? tidx % width : threadIdx.x;
    const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
    const int blockRowIdx = bidx * AGG_SHORT_ROWS_LOOPS_Y * AGG_SHORT_ROWS_THREADS_Y;
    float* shmemWrite = shmem + MUL24(ty, shmemX) + tx;
    matSum += blockRowIdx + tidx;
//    shmem[MUL24(threadIdx.y, shmemX) + threadIdx.x] = 0;
    mat += width * blockRowIdx + MUL24(ty, width) + tx;
    float* shmemWriteZeros = &shmem[MUL24(threadIdx.y,shmemX) + threadIdx.x];

    bool doSum = tidx < AGG_SHORT_ROWS_THREADS_Y ;

    if (blockRowIdx < height) {
#pragma unroll
        for (int y = 0; y < AGG_SHORT_ROWS_LOOPS_Y*AGG_SHORT_ROWS_THREADS_Y; y += AGG_SHORT_ROWS_THREADS_Y) {
//            if (y * AGG_SHORT_ROWS_THREADS_Y + idxY >= height) {
//                return; // we're done here
//            }
            doSum &= tidx + y + blockRowIdx < height;
            const bool heightIdxOK = ty < AGG_SHORT_ROWS_THREADS_Y && ty + y + blockRowIdx < height;

            shmemWriteZeros[0] = AGG_TYPE == AGG_MAX ? -2e38 : 0;

            if(AGG_TYPE == AGG_SUM) {
#pragma unroll
                for(int x = 0; x < LOOPS_X * AGG_SHORT_ROWS_THREADS_X; x+= AGG_SHORT_ROWS_THREADS_X) {
                    __syncthreads();
                    if (heightIdxOK && x + tx < width) {
                        shmemWrite[0] += mat[x];
                    }
                }
            } else {
#pragma unroll
                for(int x = 0; x < LOOPS_X * AGG_SHORT_ROWS_THREADS_X; x+= AGG_SHORT_ROWS_THREADS_X) {
                    __syncthreads();
                    if (heightIdxOK && x + tx < width) {
                        shmemWrite[0] = myMax(mat[x], shmemWrite[0]);
                    }
                }
            }
            __syncthreads();
            if (doSum) {
                /*
                 * I tried doing this final sum as a 4-step reduction, with 8 threads
                 * per warp participating. It was slightly slower.
                 */
                float accum = AGG_TYPE == AGG_MAX ? -2e38 : 0;
                float* shmemRead = shmem + MUL24(tidx, shmemX);
                // this loops too much if the rows are really short :(
#pragma unroll
                for (int i = 0; i < SUM_WIDTH_UPPERBOUND; i++) {
                    if (AGG_TYPE == AGG_MAX) {
                        accum = myMax(accum, shmemRead[0]);
                    } else if (AGG_TYPE == AGG_SUM) {
                        accum += shmemRead[0];
                    }
                    shmemRead++;
                }
                matSum[0] = accum;
                matSum += AGG_SHORT_ROWS_THREADS_Y;
            }
            __syncthreads();
            mat += width * AGG_SHORT_ROWS_THREADS_Y;
        }
    }
}

template <int AGG_TYPE>
__global__ void kAggShortRows2(float* mat, float* matSum, int width, int height) {
    const int shmemX = AGG_SHORT_ROWS_THREADS_X + 1;
    __shared__ float shmem[AGG_SHORT_ROWS_THREADS_Y*shmemX];
    const int LOOPS_X = DIVUP(width, AGG_SHORT_ROWS_THREADS_X);
    const int tidx = threadIdx.y * AGG_SHORT_ROWS_THREADS_X + threadIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
    const int blockRowIdx = bidx * AGG_SHORT_ROWS_LOOPS_Y * AGG_SHORT_ROWS_THREADS_Y;

    float* shmemWrite = shmem + MUL24(ty, shmemX) + tx;
    matSum += blockRowIdx + tidx;
//    shmem[MUL24(threadIdx.y, shmemX) + threadIdx.x] = 0;
    mat += width * blockRowIdx + MUL24(ty, width) + tx;
    float* shmemWriteZeros = &shmem[MUL24(threadIdx.y,shmemX) + threadIdx.x];
    bool doSum = tidx < AGG_SHORT_ROWS_THREADS_Y;
    if(blockRowIdx < height) {

#pragma unroll
        for (int y = 0; y < AGG_SHORT_ROWS_LOOPS_Y*AGG_SHORT_ROWS_THREADS_Y; y += AGG_SHORT_ROWS_THREADS_Y) {
//            if (y * AGG_SHORT_ROWS_THREADS_Y + idxY >= height) {
//                return; // we're done here
//            }
            doSum &= tidx + y + blockRowIdx < height;
            const bool heightIdxOK = ty + y + blockRowIdx < height;
            float accum = AGG_TYPE == AGG_MAX ? -2e38 : 0;
            shmemWriteZeros[0] = AGG_TYPE == AGG_MAX ? -2e38 : 0;

            for(int x = 0; x < LOOPS_X * AGG_SHORT_ROWS_THREADS_X; x+= AGG_SHORT_ROWS_THREADS_X) {
                __syncthreads();
                if (heightIdxOK && x + tx < width) {
                    if(AGG_TYPE == AGG_SUM) {
                        shmemWrite[0] += mat[x];
                    } else {
                        shmemWrite[0] = myMax(mat[x], shmemWrite[0]);
                    }
                }
            }

            __syncthreads();
            if (doSum) {
                float* shmemRead = shmem + MUL24(tidx, shmemX);

#pragma unroll
                for (int i = 0; i < AGG_SHORT_ROWS_THREADS_X; i++) {
                    if (AGG_TYPE == AGG_MAX) {
                        accum = myMax(accum, shmemRead[0]);
                    } else if (AGG_TYPE == AGG_SUM) {
                        accum += shmemRead[0];
                    }
                    shmemRead++;
                }

                matSum[0] = accum;
                matSum += AGG_SHORT_ROWS_THREADS_Y;
            }
            __syncthreads();
            mat += width * AGG_SHORT_ROWS_THREADS_Y;
        }
    }
}

#endif /* NVMATRIX_KERNEL_H_ */
