/*
 * conv_util.cuh
 *
 *  Created on: Nov 10, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#ifndef CONV_UTIL_CUH_
#define CONV_UTIL_CUH_
#include <nvmatrix.cuh>
#include "conv_common.cuh"

__global__ void kRotate180(float* filters, float* targets, const int filterSize);
void rotate180(NVMatrix* filters, NVMatrix* targets, bool color);
void copyInto(NVMatrix* images, NVMatrix* targets, int paddingSize, bool color);
void subsample(NVMatrix* images, NVMatrix* targets, int factor, bool avoidBankConflicts=true);
void supersample(NVMatrix* images, NVMatrix* targets, int factor);
void gridToMatrix(NVMatrix* images, NVMatrix* targets, int squareSize, bool avoidBankConflicts=true);
void matrixToGrid(NVMatrix* images, NVMatrix* targets, int squareSize, bool avoidBankConflicts=true);
void sampleMultinomial(NVMatrix* images, NVMatrix* randoms, NVMatrix* targets);

/*
 * Block size (y, x) = (n, imgSize).
 * Doesn't do bounds checking. TODO: do bounds checking!
 */
template<int factor>
__global__ void kSupersampleSlow(float* images, float* targets, const int imgSizeX, const int imgSizeY) {
    extern __shared__ float shImg[];
    const int targetImgSizeX = MUL24(imgSizeX, factor);
    const int numThreads = MUL24(blockDim.x, blockDim.y);
    int tidx = MUL24(blockDim.x, threadIdx.y) + threadIdx.x;
    images += MUL24(MUL24(imgSizeX, blockIdx.x), blockDim.y) + tidx;

    targets += MUL24(MUL24(MUL24(targetImgSizeX, blockIdx.x), blockDim.y), factor) + tidx;

    shImg[tidx] = images[0];

    __syncthreads();
//    float* myShImg = shImg + tidx / factor;
//    const int shImgInc = (blockDim.x * blockDim.y) / factor;

#pragma unroll
    for (int i = 0; i < factor*factor; i++) {
        const int x = tidx % targetImgSizeX;
        const int y = tidx / targetImgSizeX;
        const int shX = x / factor;
        const int shY = y / factor;

        targets[0] = shImg[MUL24(shY, imgSizeX) + shX];
        //        myShImg += shImgInc;

        targets += numThreads;
        tidx += numThreads;
    }
}

/*
 * Block size (y, x) = (n, imgSize). This one requires that n be divisible by
 * f^2 to allow for easy indexing into the target and source matrices. Several times
 * faster than kSupersampleSlow, which has messy indexing.
 */
template<int factor>
__global__ void kSupersampleFast(float* images, float* targets, const int imgSizeX, const int imgSizeY) {
    extern __shared__ float shImg[];
    const int targetImgSizeX = MUL24(imgSizeX, factor);
    const int numThreads = MUL24(blockDim.x, blockDim.y);
    int tidx = MUL24(blockDim.x, threadIdx.y) + threadIdx.x;
    images += MUL24(MUL24(imgSizeX, blockIdx.x), blockDim.y) + tidx;

    targets += MUL24(MUL24(MUL24(targetImgSizeX, blockIdx.x), blockDim.y), factor) + tidx;

    shImg[tidx] = images[0];

    __syncthreads();
    float* myShImg = shImg + imgSizeX * (tidx / MUL24(targetImgSizeX, factor)) + (tidx % targetImgSizeX) / factor;
    const int shImgInc = numThreads / (factor * factor);

    for (int i = 0; i < factor * factor; i++) {
        targets[0] = myShImg[0];

        myShImg += shImgInc;
        targets += numThreads;
    }
}

/*
 * Block size (y, x) = (n, imgSize). This one requires that n be divisible by
 * f to allow for easy indexing into the target and source matrices. Slightly slower (~8%)
 * than kSupersampleFast, but less restrictive on block dimensions so I use this.
 *
 * TODO: there's someting strange about this function. It seems like it should go faster
 * than it does. It has f times fewer shmem accesses than kSupersampleFast and yet that seems
 * to count for nothing...
 */
template<int factor>
__global__ void kSupersampleMedium(float* images, float* targets, const int imgSizeX, const int imgSizeY) {
    extern __shared__ float shImg[];
    const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
    const int targetImgSizeX = MUL24(imgSizeX, factor);
    const int numThreads = MUL24(blockDim.x, blockDim.y);
    int tidx = MUL24(blockDim.x, threadIdx.y) + threadIdx.x;

    const int blockRowIdx = bidx * blockDim.y;
    if(blockRowIdx >= imgSizeY) { // extra block
        return;
    }

    if (threadIdx.y + blockRowIdx < imgSizeY) {
        images += MUL24(MUL24(imgSizeX, bidx), blockDim.y) + tidx;
        shImg[tidx] = images[0];
    }
    __syncthreads();

    const bool lastBlock = blockRowIdx + blockDim.y > imgSizeY;
    targets += MUL24(MUL24(MUL24(targetImgSizeX, bidx), blockDim.y), factor)
            + MUL24(MUL24(targetImgSizeX, factor), tidx / targetImgSizeX)
            + tidx % targetImgSizeX;

    float* myShImg = shImg + MUL24(imgSizeX, tidx / targetImgSizeX) + (tidx % targetImgSizeX) / factor;
    const int shImgInc = numThreads / factor;
    const int targetsInc = (numThreads - targetImgSizeX) * factor;
    if (!lastBlock) {
//        #pragma unroll
        for (int i = 0; i < factor; i++) {
            float value = myShImg[0];
            for (int j = 0; j < factor; j++) {
                targets[0] = value;
                targets += targetImgSizeX;
            }

            myShImg += shImgInc;
            targets += targetsInc;
        }
    } else {
        const int rowsPerIter = blockDim.y / factor;
        const int rowIdx = blockRowIdx + tidx / targetImgSizeX;
        for (int row = rowIdx; row < imgSizeY; row += rowsPerIter) {
            float value = myShImg[0];
            for (int j = 0; j < factor; j++) {
                targets[0] = value;
                targets += targetImgSizeX;
            }

            myShImg += shImgInc;
            targets += targetsInc;
        }
    }
}

/*
 * This version is like kSupersampleMedium but the number of threads in the y dimension doesn't
 * have to be equal to the image size, so it will work for any image/filter sizes.
 */
template<int factor>
__global__ void kSupersampleMediumLoopy(float* images, float* targets, const int imgSizeX, const int imgSizeY) {
    extern __shared__ float shImg[];
    const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
    const int targetImgSizeX = MUL24(imgSizeX, factor);
    const int numThreads = MUL24(blockDim.x, blockDim.y);
    int tidx = MUL24(blockDim.x, threadIdx.y) + threadIdx.x;

    const int blockRowIdx = bidx * blockDim.y;
    if (blockRowIdx >= imgSizeY) { // extra block
        return;
    }
    const int targetViewSizeX = MUL24(blockDim.x, factor);
    const int targetY = tidx / targetViewSizeX;
    const int targetX = tidx % targetViewSizeX;

    const int targetImgSizeXTimesFactor = MUL24(targetImgSizeX, factor);
    // hahahh these indices are so big that you have to use 32-bit multiplication
    targets += targetImgSizeXTimesFactor * bidx * blockDim.y
            + MUL24(targetImgSizeXTimesFactor, targetY)
            + targetX;
    images += MUL24(MUL24(imgSizeX, bidx), blockDim.y)
            + MUL24(threadIdx.y, imgSizeX)
            + threadIdx.x;

    const int rowsPerIter = blockDim.y / factor;
    const int shImgInc = numThreads / factor;
    const int targetsInc = MUL24(rowsPerIter - 1, targetImgSizeXTimesFactor);

    const int iters = MIN(factor, DIVUP(imgSizeY - (blockRowIdx + targetY), rowsPerIter));
    float* shImgLoad = &shImg[tidx];
    float* myShImg2 = shImg + MUL24(blockDim.x, targetY) + targetX / factor;
    const bool load = threadIdx.y + blockRowIdx < imgSizeY;
    for (int c = 0; c < imgSizeX; c += blockDim.x) {
        if (c + blockDim.x > imgSizeX) {
            c = imgSizeX - blockDim.x; // oh what a wacky hack
        }

        __syncthreads();
        if (load) {
            shImgLoad[0] = images[c];
        }
        __syncthreads();

        float* targetWrite = targets + MUL24(c, factor);
        float* myShImg = myShImg2;
        for (int i = 0; i < iters; i++) {
            for (int j = 0; j < factor; j++) {
                targetWrite[0] = myShImg[0];
                targetWrite += targetImgSizeX;
            }

            myShImg += shImgInc;
            targetWrite += targetsInc;
        }
    }
}

/*
 * Same as kSupersampleMedium but assumes images is in col-major order.
 *
 * TODO: this code will start getting slow for very big images !
 */
template<int factor>
__global__ void kSupersampleMediumTrans(float* images, float* targets, const int imgSizeX, const int imgSizeY, const int shmemX) {
    extern __shared__ float shImg[];
    const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
    const int targetImgSizeX = MUL24(imgSizeY, factor);
    const int numThreads = MUL24(blockDim.x, blockDim.y);
    int tidx = MUL24(blockDim.x, threadIdx.y) + threadIdx.x;

    const int blockColIdx = bidx * blockDim.x; // this might be pretty huge so safer not to use mul24
    if (blockColIdx >= imgSizeX) { // extra block
        return;
    }

    if (threadIdx.x + blockColIdx < imgSizeX) {
        images += blockColIdx + threadIdx.y * imgSizeX + threadIdx.x;
        shImg[MUL24(threadIdx.x, shmemX) + threadIdx.y] = images[0]; // now transposed
    }
    __syncthreads();

    const bool lastBlock = blockColIdx + blockDim.x > imgSizeX;
    targets += MUL24(MUL24(MUL24(targetImgSizeX, bidx), blockDim.x), factor)
            + MUL24(MUL24(targetImgSizeX, factor), tidx / targetImgSizeX)
            + tidx % targetImgSizeX;

    float* myShImg = shImg + MUL24(shmemX, tidx / targetImgSizeX) + (tidx % targetImgSizeX) / factor;
    const int rowsPerIter = blockDim.x / factor;
    const int shImgInc = MUL24(rowsPerIter, shmemX);
    const int targetsInc = MUL24(numThreads - targetImgSizeX, factor);
    if (!lastBlock) {
//#pragma unroll
        for (int i = 0; i < factor; i++) {
//            const float value[] = {myShImg[0]};
            for (int j = 0; j < factor; j++) {
                targets[0] = myShImg[0];
                targets += targetImgSizeX;
            }

            myShImg += shImgInc;
            targets += targetsInc;
        }
    } else {
        const int rowIdx = blockColIdx + tidx / targetImgSizeX;
        for (int row = rowIdx; row < imgSizeX; row += rowsPerIter) {
//            float value = myShImg[0];
            for (int j = 0; j < factor; j++) {
                targets[0] = myShImg[0];
                targets += targetImgSizeX;
            }

            myShImg += shImgInc;
            targets += targetsInc;
        }
    }
}

/*
 * This version is like kSupersampleMediumTrans but the number of threads in the y dimension doesn't
 * have to be equal to the image size, so it doesn't have the problem of having too few threads
 * in the x dimension which leads to uncoalesced memory reads.
 */
template<int factor>
__global__ void kSupersampleMediumTransLoopy(float* images, float* targets, const int imgSizeX, const int imgSizeY, const int shmemX) {
    extern __shared__ float shImg[];
    const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
    const int targetImgSizeX = MUL24(imgSizeY, factor);
//    const int numThreads = MUL24(blockDim.x, blockDim.y);
    int tidx = MUL24(blockDim.x, threadIdx.y) + threadIdx.x;

    const int blockColIdx = bidx * blockDim.x; // this might be pretty huge so safer not to use mul24
    if (blockColIdx >= imgSizeX) { // extra block
        return;
    }
    const int targetViewSizeX = MUL24(blockDim.y, factor);
    const int targetY = tidx / targetViewSizeX;
    const int targetX = tidx % targetViewSizeX;

    const int targetImgSizeXTimesFactor = MUL24(targetImgSizeX, factor);
    // hahahh these indices are so big that you have to use 32-bit multiplication
    targets += targetImgSizeXTimesFactor * bidx * blockDim.x
            + MUL24(targetImgSizeXTimesFactor, targetY)
            + targetX;
    images += blockColIdx
            + threadIdx.y * imgSizeX
            + threadIdx.x;

    const int colsPerIter = blockDim.x / factor;
    const int shImgInc = MUL24(colsPerIter, shmemX);
    const int targetsInc = MUL24(colsPerIter - 1, targetImgSizeXTimesFactor);

    const int iters = MIN(factor, DIVUP(imgSizeX - (blockColIdx + targetY), colsPerIter));

    float* shImgLoad = shImg + MUL24(threadIdx.x, shmemX) + threadIdx.y;
    float* myShImg2 = shImg + MUL24(shmemX, targetY) + targetX / factor;

    const bool load = threadIdx.x + blockColIdx < imgSizeX;
    for (int r = 0; r < imgSizeY; r += blockDim.y) {
        if (r + blockDim.y > imgSizeY) {
            r = imgSizeY - blockDim.y; // oh what a wacky hack
        }
        __syncthreads();
        if (load /* && threadIdx.y + r < imgSizeY*/) {
            shImgLoad[0] = images[r * imgSizeX]; // now transposed
        }
        __syncthreads();

        float* targetWrite = targets + MUL24(r, factor);
        float* myShImg = myShImg2;

//        for (int i = blockColIdx + targetY, k = 0; k < factor && i < imgSizeX; i += colsPerIter) {
        for(int i = 0; i < iters; i++) {
            for (int j = 0; j < factor; j++) {
                targetWrite[0] = myShImg[0];
                targetWrite += targetImgSizeX;
            }

            myShImg += shImgInc;
            targetWrite += targetsInc;
//            k++;
        }
    }
}

/*
 * Block size (y, x) = (nf, I) where I = img size, f = grid square size
 * This outputs the squares in row-major order. This requires 3 more ops per thread
 * than outputting in column-major order.
 */
#define GTM_BLOCK_LOOPS_Y 32
template<int factor, bool reverse>
__global__ void kGridToMatrix(float* images, float* targets, const int imgSizeX, const int imgSizeY, const int shmemX) {
    extern __shared__ float shImg[];
    const int bidx = MUL24(blockIdx.y, gridDim.x) + blockIdx.x;

    int blockRowIdx = GTM_BLOCK_LOOPS_Y * bidx * blockDim.y;
    int tidx = MUL24(blockDim.x, threadIdx.y) + threadIdx.x;

    const int imgsOffset = imgSizeX * blockRowIdx + tidx;
    images += imgsOffset;
    targets += imgsOffset;

    const int shY = tidx / factor;
    const int shX = tidx % factor;
    float* shImgWrite = reverse ? &shImg[MUL24(shY, shmemX) + shX]
                                : &shImg[MUL24(MUL24(imgSizeX, shmemX), (threadIdx.y / factor))
                                        + MUL24(threadIdx.x, shmemX)
                                        + threadIdx.y % factor];
    float* shImgRead = reverse ? &shImg[MUL24(MUL24(imgSizeX, shmemX), (threadIdx.y / factor))
                                        + MUL24(threadIdx.x, shmemX)
                                        + threadIdx.y % factor]
                               : &shImg[MUL24(shY, shmemX) + shX];

    const int imgsInc = MUL24(blockDim.y, imgSizeX);
    for (int i = 0; i < GTM_BLOCK_LOOPS_Y; i++) {
        if (blockRowIdx >= imgSizeY) { // extra block
            return;
        }
        __syncthreads();
        if (threadIdx.y + blockRowIdx < imgSizeY) {
            shImgWrite[0] = images[0];
        }
        __syncthreads();

        if (threadIdx.y + blockRowIdx < imgSizeY) {
            targets[0] = shImgRead[0];
        }

        blockRowIdx += blockDim.y;
        images += imgsInc;
        targets += imgsInc;
    }
}

#define GTM_LOOPY_BLOCK_LOOPS_Y 16
/*
 * Uses 14 registers
 */
template<int factor, bool reverse>
__global__ void kGridToMatrixLoopy(float* images, float* targets, const int imgSizeX, const int imgSizeY, const int shmemX) {
    extern __shared__ float shImg[];
    const int bidx = MUL24(blockIdx.y, gridDim.x) + blockIdx.x;

    int blockRowIdx = GTM_LOOPY_BLOCK_LOOPS_Y * bidx * blockDim.y;
    if (blockRowIdx >= imgSizeY) { // extra block
        return;
    }

    int tidx = MUL24(blockDim.x, threadIdx.y) + threadIdx.x;
    const int imgsOffset = imgSizeX * blockRowIdx;
    if (reverse) {
        targets += imgsOffset
                + MUL24(threadIdx.y, imgSizeX)
                + threadIdx.x;
        images += imgsOffset
                + MUL24(MUL24(tidx / (factor * blockDim.x), factor), imgSizeX)
                + tidx % (factor * blockDim.x);
    } else {
        images += imgsOffset
                + MUL24(threadIdx.y, imgSizeX)
                + threadIdx.x;
        targets += imgsOffset
                + MUL24(MUL24(tidx / (factor * blockDim.x), factor), imgSizeX)
                + tidx % (factor * blockDim.x);
    }

    const int shY = tidx / factor;
    const int shX = tidx % factor;
    float* shImgWrite = reverse ? &shImg[MUL24(shY, shmemX) + shX]
                                : &shImg[MUL24(MUL24(blockDim.x, shmemX), (threadIdx.y / factor))
                                        + MUL24(threadIdx.x, shmemX)
                                        + threadIdx.y % factor];
    float* shImgRead = reverse ? &shImg[MUL24(MUL24(blockDim.x, shmemX), (threadIdx.y / factor))
                                        + MUL24(threadIdx.x, shmemX)
                                        + threadIdx.y % factor]
                               : &shImg[MUL24(shY, shmemX) + shX];

    const int imgsInc = MUL24(blockDim.y, imgSizeX);
    for (int x = 0; x < imgSizeX; x += blockDim.x) {
        if (x + blockDim.x > imgSizeX) {
            x = imgSizeX - blockDim.x; // yea
        }

        float* targetsWrite = targets + (reverse ? x : MUL24(x, factor));
        float* imagesRead = images + (reverse ? MUL24(x, factor) : x);

        blockRowIdx = GTM_LOOPY_BLOCK_LOOPS_Y * bidx * blockDim.y;

        for (int y = 0; y < GTM_LOOPY_BLOCK_LOOPS_Y; y++) {
            if (blockRowIdx >= imgSizeY) { // extra block
                break;
            }
            __syncthreads();
            if (threadIdx.y + blockRowIdx < imgSizeY) {
                shImgWrite[0] = imagesRead[0];
            }
            __syncthreads();

            if (threadIdx.y + blockRowIdx < imgSizeY) {
                targetsWrite[0] = shImgRead[0];
            }

            blockRowIdx += blockDim.y;
            imagesRead += imgsInc;
            targetsWrite += imgsInc;
        }
    }
}

/*
 * Factor must divide imgSize.
 * This routine is good when the subsampling region (factor) is pretty small (4x4 works well).
 * For large factors, it's inefficient but works. But my bet here is that for convolutional nets,
 * typically we won't want to subsample by a factor of more than 8 or so.
 *
 * Each sum of f^2 elements is computed by f cooperating threads. It works better than using
 * reduction in most (or all? I don't remember) the cases I've tried. One of the problems with reductions
 * is that if the number of elements you want to sum is not a power of 2, you have to do
 * a lot of bounds checking.
 *
 * Block size (y,x) = (regionsYPerBlock, imgSize).
 */
template<int factor, bool checkThreadBounds>
__global__ void kSubsample_noreduc(float* images, float* targets, const int imgSize, const int numRegionsY, const int shmemX) {
    extern __shared__ float shImg[];

    const int regionsYPerBlock = blockDim.y;
    const int bidx = MUL24(blockIdx.y, gridDim.x) + blockIdx.x;
    const int blockRegionIdxY = MUL24(regionsYPerBlock, bidx);

    if (blockRegionIdxY >= numRegionsY) {
        return;
    }

    const int tidx = MUL24(threadIdx.y, blockDim.x) + threadIdx.x;
    const int numRegionsX = imgSize / factor;
    const int regionPixels = factor * factor;
    const int regionsPerBlock = MUL24(numRegionsX, regionsYPerBlock);
    const int blockRegionIdx = MUL24(regionsPerBlock, bidx);
    const int threadRegionIdxY = blockRegionIdxY + threadIdx.y;
    const int regionsInThisBlock = numRegionsY - blockRegionIdxY < regionsYPerBlock
                                    ? MUL24(numRegionsX, numRegionsY - blockRegionIdxY) : regionsPerBlock;

    float* myShImg = shImg + MUL24((threadIdx.x % factor), shmemX) + (threadIdx.x / factor) + MUL24(threadIdx.y, numRegionsX);
    if (!checkThreadBounds || threadRegionIdxY < numRegionsY) {

        images += MUL24(MUL24(threadIdx.y, factor), imgSize)
                + MUL24(blockRegionIdx, regionPixels)
                + threadIdx.x;

        float mySum = 0;
        for (int d = 0; d < factor; d++) {
            mySum += images[0];
            images += imgSize;
        }
        myShImg[0] = mySum; // conflicts perhaps
    }

    __syncthreads();
    // Now sum out cols of shImg
    if (tidx < regionsInThisBlock) { // no bank conflicts
        float mySum = 0;
        myShImg = shImg + tidx;
        for (int d = 0; d < factor; d++) {
            mySum += myShImg[0];
            myShImg += shmemX;
        }
        targets[blockRegionIdx + tidx] = mySum / regionPixels;
    }
}

/*
 * This is just like the above but with a reduction at the end.
 * Will fail if factor not power of 2.
 */
template<int factor, bool checkThreadBounds>
__global__ void kSubsample_reduc2(float* images, float* targets, const int imgSize, const int numRegionsY, const int shmemX) {
    extern __shared__ float shImg[];

    const int regionsYPerBlock = blockDim.y;
    const int bidx = MUL24(blockIdx.y, gridDim.x) + blockIdx.x;
    const int blockRegionIdxY = MUL24(regionsYPerBlock, bidx);

    if (blockRegionIdxY >= numRegionsY) {
        return;
    }

    const int tidx = MUL24(threadIdx.y, blockDim.x) + threadIdx.x;
    const int numRegionsX = imgSize / factor;
    const int regionPixels = factor * factor;
    const int regionsPerBlock = MUL24(numRegionsX, regionsYPerBlock);
    const int blockRegionIdx = MUL24(regionsPerBlock, bidx);
    const int threadRegionIdxY = blockRegionIdxY + threadIdx.y;
    const int regionsInThisBlock = numRegionsY - blockRegionIdxY < regionsYPerBlock
                                    ? MUL24(numRegionsX, numRegionsY - blockRegionIdxY) : regionsPerBlock;

    float* myShImg = shImg + MUL24((threadIdx.x % factor), shmemX) + (threadIdx.x / factor) + MUL24(threadIdx.y, numRegionsX);
    if (!checkThreadBounds || threadRegionIdxY < numRegionsY) {

        images += MUL24(MUL24(threadIdx.y, factor), imgSize) + MUL24(blockRegionIdx, regionPixels) + threadIdx.x;

        float mySum = 0;
        for (int d = 0; d < factor; d++) {
            mySum += images[0];
            images += imgSize;
        }
        myShImg[0] = mySum; // conflicts perhaps
    }

    __syncthreads();

    const int tx = tidx % regionsInThisBlock;
    const int ty = tidx / regionsInThisBlock;
    myShImg = shImg + tidx;
    for (int d = factor / 2; d > 0; d /= 2) {

        if (ty < d) {
            myShImg[0] += myShImg[d * shmemX];
        }
        __syncthreads();
    }
    if (ty == 0) {
        targets[blockRegionIdx + tx] = myShImg[0];
    }
}

template<int factor>
__global__ void kSubsample_reduc(float* images, float* targets, const int imgSize, const int numRegionsY) {
    extern __shared__ float shImg[]; // blockDim.x * blockDim.y

    const int tidx = threadIdx.y * blockDim.x + threadIdx.x;

    const int numRegionsX = imgSize / factor;
    const int regionPixels = factor*factor;
    const int regionsPerBlock = numRegionsX * (blockDim.y / (factor/2));
    const int row = threadIdx.y % (factor / 2);
//    const int threadsPerRegionRow = blockDim.x*(factor/2);

//    const int regionIdxX = (tidx / factor) % factor;
//    const int regionIdxY = threadIdx.y / (factor/2);
//    const int blockRegionIdx = blockIdx.x * regionsPerBlock;
//    const int regionIdx = blockRegionIdx + tidx / (imgSize*(imgSize / factor));

    images += MUL24(MUL24((threadIdx.y / (factor / 2)), factor), imgSize) + MUL24(row, imgSize) + threadIdx.x + MUL24(MUL24(blockIdx.x, regionsPerBlock), regionPixels);


//    const int shY = tidx / (regionsPerBlock*factor);
//    const int shX = tidx % (regionsPerBlock*factor);
//    if (blockIdx.x > 1) {
//        return;
//    }

    float* myShImg = shImg + tidx;
    myShImg[0] = images[0] + images[MUL24((factor / 2), imgSize)];

    for (int d = factor / 4; d > 0; d /= 2) { // col-wise summation, no bank conflicts
        __syncthreads();
        if (row < d) {
            myShImg[0] += myShImg[MUL24(d, imgSize)];
        }
    }

    if (factor % 2 == 1) { // odd factor size :(
        // do something esle here too
    }

    if(factor <= 16) {
        __syncthreads();
        if (row == 0) {
#pragma unroll
            for (int d = factor / 2; d > 0; d /= 2) {
                myShImg[0] += myShImg[d];
            }
        }
    } else {
#pragma unroll
        for (int d = factor / 2; d > 0; d /= 2) {
            __syncthreads();
            if (row == 0) {
                myShImg[0] += myShImg[d];
            }
        }
    }
    if (factor % 2 == 1) {
        // do something esle here too
    }
    __syncthreads();
    if (tidx < regionsPerBlock) {
        targets += blockIdx.x*regionsPerBlock + tidx;
        // bank conflicts here
        targets[0] = shImg[MUL24((tidx % numRegionsX), factor) + MUL24(MUL24((tidx / numRegionsX), (factor / 2)), blockDim.x)];
    }
}

/*
 * The compiler is supposed to convert i / 16 to i >> 4, so
 * why is i >> 4 still so much faster?
 */
#define IDX(i) ((i) + ((i) >> 4))
//#define IDX(i) (i)

/*
 * Samples from a bunch of multinomial distributions, where each row of data
 * is a different distribution.
 *
 * Uses the scan algorithm from these slides http://www.eecg.toronto.edu/~moshovos/CUDA08/slides/007%20-%20Scans.ppt
 * to compute prefix-sums.
 */
template<int bX>
__global__ void kSampleMultinomial(float* data, float* randoms, float* targets, const int multiSize, const int numMulti)  {
    const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
    data += multiSize * bidx  + threadIdx.x;
    targets += multiSize * bidx  + threadIdx.x;
    __shared__ float shmem[IDX(bX * 2 + 1)];
    __shared__ float rand;

    if (bidx >= numMulti)
        return;

    shmem[IDX(threadIdx.x)] = 0;
    shmem[IDX(threadIdx.x + bX)] = 0;
    if (threadIdx.x < multiSize) {
        shmem[IDX(threadIdx.x)] = data[0]; // load input into shared memory
        if (threadIdx.x + bX < multiSize) {
            shmem[IDX(threadIdx.x + bX)] = data[bX];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        rand = randoms[bidx];
    }
    /*=============================================================
     * Reduction
     */
    int ai = 2 * threadIdx.x;
    int bi = ai + 1;
    if (bX >= 512) {
        __syncthreads();

        shmem[IDX(bi)] += shmem[IDX(ai)];

        ai = (ai << 1) + 1;
        bi = (bi << 1) + 1;
    }

    if (bX >= 256) {
        __syncthreads();
        if (threadIdx.x < 256) {
            shmem[IDX(bi)] += shmem[IDX(ai)];
        }
        ai = (ai << 1) + 1;
        bi = (bi << 1) + 1;
    }

    if (bX >= 128) {
        __syncthreads();
        if (threadIdx.x < 128) {
            shmem[IDX(bi)] += shmem[IDX(ai)];
        }
        ai = (ai << 1) + 1;
        bi = (bi << 1) + 1;
    }

    if (bX >= 64) {
        __syncthreads();
        if (threadIdx.x < 64) {
            shmem[IDX(bi)] += shmem[IDX(ai)];
        }
        ai = (ai << 1) + 1;
        bi = (bi << 1) + 1;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        shmem[IDX(bi)] += shmem[IDX(ai)];
    }
    ai = (ai << 1) + 1;
    bi = (bi << 1) + 1;

    if (threadIdx.x < 16) {
        shmem[IDX(bi)] += shmem[IDX(ai)];
    }
    ai = (ai << 1) + 1;
    bi = (bi << 1) + 1;

    if (threadIdx.x < 8) {
        shmem[IDX(bi)] += shmem[IDX(ai)];
    }
    ai = (ai << 1) + 1;
    bi = (bi << 1) + 1;

    if (threadIdx.x < 4) {
        shmem[IDX(bi)] += shmem[IDX(ai)];
    }
    ai = (ai << 1) + 1;
    bi = (bi << 1) + 1;

    if (threadIdx.x < 2) {
        shmem[IDX(bi)] += shmem[IDX(ai)];
    }
    ai = (ai << 1) + 1;
    bi = (bi << 1) + 1;

    if (threadIdx.x < 1) {
        shmem[IDX(bi)] += shmem[IDX(ai)];

        /*=============================================================
         * Scan
         */
        shmem[IDX(bX * 2 - 1)] = 0;

        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }
    ai >>= 1;
    bi >>= 1;

    if (threadIdx.x < 2) {
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }
    ai >>= 1;
    bi >>= 1;

    if (threadIdx.x < 4) {
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }
    ai >>= 1;
    bi >>= 1;

    if (threadIdx.x < 8) {
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }
    ai >>= 1;
    bi >>= 1;

    if (threadIdx.x < 16) {
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }
    ai >>= 1;
    bi >>= 1;

    if (threadIdx.x < 32) {
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }

    if (bX >= 64) {
        __syncthreads();
        ai >>= 1;
        bi >>= 1;
        if (threadIdx.x < 64) {
            const float t = shmem[IDX(ai)];
            shmem[IDX(ai)] = shmem[IDX(bi)];
            shmem[IDX(bi)] += t;
        }
    }

    if (bX >= 128) {
        __syncthreads();
        ai >>= 1;
        bi >>= 1;
        if (threadIdx.x < 128) {
            const float t = shmem[IDX(ai)];
            shmem[IDX(ai)] = shmem[IDX(bi)];
            shmem[IDX(bi)] += t;
        }
    }

    if (bX >= 256) {
        __syncthreads();
        ai >>= 1;
        bi >>= 1;
        if (threadIdx.x < 256) {
            const float t = shmem[IDX(ai)];
            shmem[IDX(ai)] = shmem[IDX(bi)];
            shmem[IDX(bi)] += t;
        }
    }

    if (bX >= 512) {
        __syncthreads();
        ai >>= 1;
        bi >>= 1;
        const float t = shmem[IDX(ai)];
        shmem[IDX(ai)] = shmem[IDX(bi)];
        shmem[IDX(bi)] += t;
    }

    __syncthreads();
    if (threadIdx.x < multiSize) {
        shmem[IDX(threadIdx.x)] += data[0]; // load input into shared memory
        if (threadIdx.x + bX < multiSize) {
            shmem[IDX(threadIdx.x + bX)] += data[bX];
        }
    }
    __syncthreads();
    if (threadIdx.x < multiSize) {
        const float prev = threadIdx.x == 0 ? 0 : shmem[IDX(threadIdx.x - 1)];
        targets[0] = rand >= prev && rand < shmem[IDX(threadIdx.x)];
//        targets[0] = shmem[IDX(threadIdx.x)];

        if (threadIdx.x + bX < multiSize) {
            targets[bX] = rand >= shmem[IDX(threadIdx.x - 1 + bX)] && rand < shmem[IDX(threadIdx.x + bX)];
//            targets[bX] = shmem[IDX(threadIdx.x + bX)];
        }
    }
}
#define SSM_THREADS_X   16
#define SSM_THREADS_Y   32
#define SSM_LOOPS_Y     16
/*
 * This routine is just always faster than the fancy tree-based one above...
 * Oh ok, not in all cases. In the cases when the number of distributions
 * that you want to sample from (height) is fairly large.
 *
 * TODO: revisit this routine cause that doWrite statement is too long
 * and it all can probably be simplified if i control the block size at run-time
 */
template <int LOOPS_X, int SUM_WIDTH_UPPERBOUND>
__global__ void kSampleSmallMultinomial(float* multi, float* randoms, float* targets, const int width, const int height) {
    const int shmemX = SSM_THREADS_X + 1;
    __shared__ float shmem[SSM_THREADS_Y*shmemX];

//    const int LOOPS_X = DIVUP(width, AGG_SHORT_ROWS_THREADS_X);

    const int bidx = blockIdx.y * gridDim.x + blockIdx.x;
    const int blockRowIdx = bidx * SSM_LOOPS_Y * SSM_THREADS_Y;

    if(blockRowIdx < height) {
        const int tidx = threadIdx.y * SSM_THREADS_X + threadIdx.x;
        int ty = LOOPS_X == 1 ? tidx / width : threadIdx.y;
        const int tx = LOOPS_X == 1 ? tidx % width : threadIdx.x;
        float* shmemWrite = shmem + MUL24(ty, shmemX) + tx;
        //    targets += blockIdx.y * AGG_SHORT_ROWS_LOOPS_Y * AGG_SHORT_ROWS_THREADS_Y + tidx;
        const int dataOffset = width * blockRowIdx + MUL24(ty, width) + tx;
        multi += dataOffset;
        targets += dataOffset;

        float* shmemWriteZeros = &shmem[MUL24(threadIdx.y,shmemX) + threadIdx.x];
//        ty += blockRowIdx;
//#pragma unroll
        for (int y = 0; y < SSM_LOOPS_Y*SSM_THREADS_Y; y += SSM_THREADS_Y) {
//            if (y * AGG_SHORT_ROWS_THREADS_Y + idxY >= height) {
//                return; // we're done here
//            }
            const bool doSum = tidx < SSM_THREADS_Y && tidx + y + blockRowIdx < height;
            float rnd;
            if (doSum) {
                rnd = randoms[tidx + y + blockRowIdx];
            }
            float accum = 0, accumPrev = 0;
//#pragma unroll // this causes > 16 registers to be used in some cases, avoid
            for(int x = 0; x < LOOPS_X * SSM_THREADS_X; x+= SSM_THREADS_X) {
                __syncthreads();
                shmemWriteZeros[0] = 0;
                if (LOOPS_X == 1) { // because the part we zeroed might not be same as one we're writing to
                    __syncthreads();
                }
                const bool doWrite = ty + blockRowIdx + y < height && (LOOPS_X > 1 || ty < SSM_THREADS_Y) && x + tx < width;
                if (doWrite) {
                    shmemWrite[0] = multi[y * width + x];
                }
                __syncthreads();

                if (doSum) {
                    float* shmemRead = shmem + MUL24(tidx, shmemX);

                    // this loops too much if the rows are really short :(
                    for (int i = 0; i < SUM_WIDTH_UPPERBOUND; i++) {
                        accumPrev = accum;
                        accum += shmemRead[0];
                        shmemRead[0] = rnd >= accumPrev && rnd < accum;
                        shmemRead++;
                    }
                }
                __syncthreads();
                if (doWrite) {
                    targets[y * width + x] = shmemWrite[0];
                }
            }
//            multi += width * SSM_THREADS_Y;
//            targets += width * SSM_THREADS_Y;
//            ty += SSM_THREADS_Y;
        }
    }
}

#endif /* CONV_UTIL_CUH_ */
