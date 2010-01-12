/*
 * conv_util.cu
 *
 *  Created on: Nov 10, 2009
 *      Author: Alex Krizhevsky
 *
 *  These are routines that are useful for convolutional neural nets/RBMs.
 */

#include <cutil_inline.h>
#include <assert.h>
#include "conv_util.cuh"
#include "conv_common.cuh"

/*
 * Block size 16x16
 * Don't need shared memory on devices with compute capability 1.3 because memory
 * doesn't have to be accessed sequentially by threads.
 *
 * This is far from perfect, and in many cases is actually slwoer than doing it on the
 * CPU but still this takes so little time that it doesn't matter.
 */
__global__ void kRotate180(float* filters, float* targets, const int filterSize) {
//   __shared__ float shFilter[16][16];

    const int filtIdx = blockIdx.x;
    const int readStart = MUL24(MUL24(filterSize, filterSize), filtIdx);
    filters += readStart;
    targets += readStart;

    for(int y = threadIdx.y; y < filterSize; y += 16) {
        for(int x = threadIdx.x; x < filterSize; x += 16) {
            const int writeX = filterSize - 1 - x;
            const int writeY = filterSize - 1 - y;

            targets[MUL24(writeY, filterSize) + writeX] = filters[MUL24(y, filterSize) + x];
        }
    }
}

/*
 * Block size 16x16.
 * Probably a better idea to allocate multiple blocks per image so you don't have
 * to loop inside the block.
 */
__global__ void kCopyInto(float* images, float* targets, const int imgSize, const int paddingSize, const int numImages) {
    const int imgIdx = blockIdx.y * gridDim.x + blockIdx.x;
    if (imgIdx < numImages) {
        const int targetSize = imgSize + 2 * paddingSize;
        images += imgIdx * imgSize * imgSize;
        targets += imgIdx * targetSize * targetSize + MUL24(paddingSize, targetSize) + paddingSize;
        for (int y = threadIdx.y; y < imgSize; y += 16) {
            for (int x = threadIdx.x; x < imgSize; x += 16) {
                targets[MUL24(y, targetSize) + x] = images[MUL24(y, imgSize) + x];
            }
        }
    }
}

void rotate180(NVMatrix* filters, NVMatrix* targets, bool color=false) {
    assert(!color || filters->getNumCols() % 3 == 0);

    assert(!color && floor(sqrt(float(filters->getNumCols()))) == sqrt(float(filters->getNumCols()))
            || color && floor(sqrt(float(filters->getNumCols() / 3))) == sqrt(float(filters->getNumCols() / 3)));
    assert(targets->isSameDims(*filters));
    int numFilters =  (color ? 3 : 1) * filters->getNumRows();
    int filterSize = color ? int(sqrt(filters->getNumCols() / 3)) : int(sqrt(filters->getNumCols()));
    dim3 threads(16, 16, 1);
    dim3 blocks(numFilters, 1, 1);
    kRotate180<<<blocks, threads>>>(filters->getDevData(), targets->getDevData(), filterSize);
    cutilCheckMsg("kernel execution failed");
}

/*
 * This function copies the images in "images" into "targets" and adds a padding.
 *
 * Specifically, suppose "images" contains just one image and it looks like this:
 * IIII
 * IIII
 * IIII
 *
 * And targets looks like this:
 * XXXXXX
 * XXXXXX
 * XXXXXX
 * XXXXXX
 * XXXXXX
 *
 * After this function is called, targets will look like this:
 * XXXXXX
 * XIIIIX
 * XIIIIX
 * XIIIIX
 * XXXXXX
 *
 * Where the Is and Xs are arbitrary values.
 *
 * You can use this function to pad a bunch of images with a border of zeros. To do this,
 * the targets matrix should be all zeros.
 *
 */
void copyInto(NVMatrix* images, NVMatrix* targets, int paddingSize, bool color=false) {
    assert(!color || images->getNumCols() % 3 == 0);

    assert(!color && floor(sqrt(float(images->getNumCols()))) == sqrt(float(images->getNumCols()))
            || color && floor(sqrt(float(images->getNumCols() / 3))) == sqrt(float(images->getNumCols() / 3)));
    int imgSize = color ? int(sqrt(images->getNumCols() / 3)) : int(sqrt(images->getNumCols()));
    int numImages =  (color ? 3 : 1) * images->getNumRows();
    assert(targets->getNumElements() == numImages * (imgSize + 2*paddingSize)*(imgSize + 2*paddingSize));

    dim3 threads(16, 16, 1);
    dim3 blocks(numImages, 1, 1);
    while(blocks.x > NUM_BLOCKS_MAX) {
        blocks.x = DIVUP(blocks.x, 2);
        blocks.y *= 2;
    }
    kCopyInto<<<blocks, threads>>>(images->getDevData(), targets->getDevData(), imgSize, paddingSize, numImages);
    cutilCheckMsg("kernel execution failed");
}

/*
 * f = factor, m = image size
 * Converts a bunch of mxm images to (m/f)x(m/f) images by averaging non-overlapping fxf regions.
 *
 * The avoidBankConflicts option causes this function to use extra shared memory to avoid all
 * bank conflicts. Most bank conflicts are avoided regardless of the setting of this parameter,
 * and so setting this parameter to true will have minimal impact on performance (I noticed
 * a 5% improvement). (stil can get 2-way conflicts if factor doesn't divide 16)
 */
void subsample(NVMatrix* images, NVMatrix* targets, int factor, bool avoidBankConflicts) {
    int imgPixels = images->getNumCols();
    assert(sqrt(float(imgPixels)) == floor(sqrt(float(imgPixels))));
    int imgSize = sqrt(imgPixels);
    assert(imgSize > factor);
    assert(imgSize % factor == 0);
    assert(factor <= 16);
    assert(factor >= 2);
    assert(imgSize <= 512);
//    assert(factor % 2 == 0); // TODO: remove this restriction
    int numRegions = images->getNumElements() / (factor*factor);
    int numRegionsY = (imgSize / factor) * images->getNumRows();

    assert(targets->getNumElements() == numRegions);
//    assert(imgSize * (factor/2) <= 512); // for now
    int regionsXPerBlock = imgSize / factor;
    int numThreadsX = imgSize;
    int SHMEM_MAX = 8192; // don't use more than this much shmem
    int regionsYPerBlock = MIN(512 / numThreadsX, SHMEM_MAX / (4*imgSize)); // to avoid running out of shmem
//    regionsYPerBlock--;
    int regionsPerBlock = regionsYPerBlock * regionsXPerBlock;

    // this will avoid all bank conflicts but may (?) use up too much shmem
    int shmemPadX = avoidBankConflicts * (DIVUP(16,factor) + (regionsPerBlock % 16 == 0 ? 0 : 16 - regionsPerBlock % 16));
//    shmemPadX = 0;
    int shmemY = factor, shmemX = regionsPerBlock + shmemPadX;
    int shmem = 4 * shmemX * shmemY;
    if (shmem == 0 || shmem > 16300) {
        // this really shouldn't happen and i've only put this here as a precautionary measure
        // to avoid getting mysteriously wrong results.
        fprintf(stderr, "subsample: not enough shared memory!");
        exit(EXIT_FAILURE);
    }

    int numThreadsY = regionsYPerBlock;
//    int blocks = numRegionsY / regionsYPerBlock;
    int blocksX = imgSize / factor, blocksY = DIVUP(images->getNumRows(), regionsYPerBlock);
    assert(blocksX < 65535 && blocksY < 65535);
//    assert(numRegionsY % regionsYPerBlock == 0);
    bool checkThreadBounds = numRegionsY % regionsYPerBlock != 0;
//    printf("num regions y: %d, regions y per block: %d\n", numRegionsY, regionsYPerBlock);
    dim3 grid(blocksX, blocksY);
    dim3 threads(numThreadsX, numThreadsY);
//    printf("grid: %ux%u, threads: %ux%u\n", grid.y, grid.x, threads.y, threads.x);
//    printf("check bounds: %u\n", checkThreadBounds);
//    printf("using %u bytes of shmem\n", shmem);
    if (factor == 2) {
        if (checkThreadBounds) {
            kSubsample_noreduc<2, true><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        } else {
            kSubsample_noreduc<2, false><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        }
    } else if (factor == 3) {
        if (checkThreadBounds) {
            kSubsample_noreduc<3, true><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        } else {
            kSubsample_noreduc<3, false><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        }
    } else if (factor == 4) {
        if (checkThreadBounds) {
            kSubsample_noreduc<4, true><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        } else {
            kSubsample_noreduc<4, false><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        }
    } else if (factor == 5) {
        if (checkThreadBounds) {
            kSubsample_noreduc<5, true><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        } else {
            kSubsample_noreduc<5, false><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        }
    } else if (factor == 6) {
        if (checkThreadBounds) {
            kSubsample_noreduc<6, true><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        } else {
            kSubsample_noreduc<6, false><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        }
    } else if (factor == 7) {
        if (checkThreadBounds) {
            kSubsample_noreduc<7, true><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        } else {
            kSubsample_noreduc<7, false><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        }
    } else if (factor == 8) {
        if (checkThreadBounds) {
            kSubsample_noreduc<8, true><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        } else {
            kSubsample_noreduc<8, false><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        }
    } else if (factor == 9) {
        if (checkThreadBounds) {
            kSubsample_noreduc<9, true><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        } else {
            kSubsample_noreduc<9, false><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        }
    } else if (factor == 10) {
        if (checkThreadBounds) {
            kSubsample_noreduc<10, true><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        } else {
            kSubsample_noreduc<10, false><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        }
    } else if (factor == 11) {
        if (checkThreadBounds) {
            kSubsample_noreduc<11, true><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        } else {
            kSubsample_noreduc<11, false><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        }
    } else if (factor == 12) {
        if (checkThreadBounds) {
            kSubsample_noreduc<12, true><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        } else {
            kSubsample_noreduc<12, false><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        }
    } else if (factor == 13) {
        if (checkThreadBounds) {
            kSubsample_noreduc<13, true><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        } else {
            kSubsample_noreduc<13, false><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        }
    } else if (factor == 14) {
        if (checkThreadBounds) {
            kSubsample_noreduc<14, true><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        } else {
            kSubsample_noreduc<14, false><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        }
    } else if (factor == 15) {
        if (checkThreadBounds) {
            kSubsample_noreduc<15, true><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        } else {
            kSubsample_noreduc<15, false><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        }
    } else if (factor == 16) {
        if (checkThreadBounds) {
            kSubsample_noreduc<16, true><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        } else {
            kSubsample_noreduc<16, false><<<grid, threads,shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY, shmemX);
        }
    }
    cutilCheckMsg("kernel execution failed");

//    if(factor == 4) {
////        kSubsample_reduc<4><<<grid, threads,4*numThreadsX*numThreadsY>>>(images->getDevData(), targets->getDevData(), imgSize, numRegionsY);
//    }
}

/*
 * This is kind of a mess...could use some cleanup.
 * Blows up a bunch of mxm images to (mf)x(mf)
 */
void supersample(NVMatrix* images, NVMatrix* targets, int factor) {
    bool trans = images->isTrans();
    int imgPixels = images->getNumCols();
    int numImages = images->getNumRows();
    assert(sqrt(float(imgPixels)) == floor(sqrt(float(imgPixels))));
    int imgSize = sqrt(imgPixels);
    assert(factor > 1 && factor <= 16);
    assert(imgSize > 0 && imgSize <= 512);

    int targetPixels = targets->getNumCols();
    assert(sqrt(float(targetPixels)) == floor(sqrt(float(targetPixels))));
    int targetSize = sqrt(targetPixels);
    assert(targetSize % factor == 0);
    assert(targetSize / factor == imgSize);
    assert(targets->getNumElements() == images->getNumElements() * factor*factor);

    int threadsX, threadsY;
    int SHMEM_MAX = 8192; // don't use more than this much shmem
    int shmemX, shmemY, blocksX, blocksY;
    bool useLoopy = false;
    int THREADS_MAX_LOOPY = 512, THREADS_MAX = trans ? 256 : 512;
    if (!trans) {
        threadsX = imgSize;
        threadsY = factor * MIN(THREADS_MAX / (factor*threadsX), SHMEM_MAX / (4*threadsX*factor)); // to avoid running out of shmem

        if(threadsY == 0) {
            assert(factor <= 32); // yes this is covered by assert above but in case i ever remove that
            THREADS_MAX = 512;
            useLoopy = true;
            threadsX = MIN(16, imgSize); // not that imgsize can be < 16 here under current conditions
            threadsY = factor * MIN(THREADS_MAX_LOOPY / (factor*threadsX), SHMEM_MAX / (4*threadsX*factor)); // to avoid running out of shmem
        }

        shmemY = threadsY;
        shmemX = threadsX;
        blocksX = imgSize;
        blocksY = DIVUP(numImages, threadsY);
//        printf("boundary problems: %u\n", numImages % threadsY != 0);
    } else {

        threadsY = imgSize;
        threadsX = factor * MIN(THREADS_MAX / (factor*threadsY), SHMEM_MAX / (4*threadsY*factor)); // to avoid running out of shmem

        if(threadsX < 8) {
            useLoopy = true;
            int xFactorMult = DIVUP(16, factor);
            threadsX = xFactorMult * factor;
            threadsY = THREADS_MAX / threadsX;
            int newThreadsX = threadsX, newThreadsY = threadsY;
            while (newThreadsY > 0 && imgSize % newThreadsY != 0) { // let's see if we can make threadsY divide imgSize
                newThreadsX += factor;
                newThreadsY = THREADS_MAX / newThreadsX;
            }
            if (newThreadsY > 0) {
                threadsY = newThreadsY;
                threadsX = newThreadsX;
            }

            assert(threadsY > 0);
        }

        shmemY = threadsX;
        shmemX = threadsY + (1 - (threadsY % 2));
        blocksX = DIVUP(numImages, threadsX);
        blocksY = imgSize;
//        printf("boundary problems: %u\n", numImages % threadsX != 0);
    }
    int shmem = 4 * shmemX * shmemY;
    if (shmem == 0 || shmem > 16300) {
        // this really shouldn't happen and i've only put this here as a precautionary measure
        // to avoid getting mysteriously wrong results.
        fprintf(stderr, "supersample: not enough shared memory!");
        exit(EXIT_FAILURE);
    }

    dim3 grid(blocksX, blocksY);
    dim3 threads(threadsX, threadsY);
//    printf("blocks: %dx%d, threads: %dx%d\n", blocksY, blocksX, threadsY, threadsX);
//    printf("using %dx%d = %d bytes of shmem\n", shmemY, shmemX, shmem);

    if(!trans) {
        if(!useLoopy) {
            if(factor == 2) {
                kSupersampleMedium<2><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 3) {
                kSupersampleMedium<3><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 4) {
                kSupersampleMedium<4><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 5) {
                kSupersampleMedium<5><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 6) {
                kSupersampleMedium<6><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 7) {
                kSupersampleMedium<7><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 8) {
                kSupersampleMedium<8><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 9) {
                kSupersampleMedium<9><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 10) {
                kSupersampleMedium<10><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 11) {
                kSupersampleMedium<11><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 12) {
                kSupersampleMedium<12><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 13) {
                kSupersampleMedium<13><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 14) {
                kSupersampleMedium<14><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 15) {
                kSupersampleMedium<15><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 16) {
                kSupersampleMedium<16><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            }
        } else {
            if(factor == 2) {
                kSupersampleMediumLoopy<2><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 3) {
                kSupersampleMediumLoopy<3><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 4) {
                kSupersampleMediumLoopy<4><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 5) {
                kSupersampleMediumLoopy<5><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 6) {
                kSupersampleMediumLoopy<6><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 7) {
                kSupersampleMediumLoopy<7><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 8) {
                kSupersampleMediumLoopy<8><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 9) {
                kSupersampleMediumLoopy<9><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 10) {
                kSupersampleMediumLoopy<10><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 11) {
                kSupersampleMediumLoopy<11><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 12) {
                kSupersampleMediumLoopy<12><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 13) {
                kSupersampleMediumLoopy<13><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 14) {
                kSupersampleMediumLoopy<14><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 15) {
                kSupersampleMediumLoopy<15><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            } else if(factor == 16) {
                kSupersampleMediumLoopy<16><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize);
            }
        }
    } else {
        if(!useLoopy) {
            if(factor == 2) {
                kSupersampleMediumTrans<2><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 3) {
                kSupersampleMediumTrans<3><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 4) {
                kSupersampleMediumTrans<4><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 5) {
                kSupersampleMediumTrans<5><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 6) {
                kSupersampleMediumTrans<6><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 7) {
                kSupersampleMediumTrans<7><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 8) {
                kSupersampleMediumTrans<8><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 9) {
                kSupersampleMediumTrans<9><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 10) {
                kSupersampleMediumTrans<10><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 11) {
                kSupersampleMediumTrans<11><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 12) {
                kSupersampleMediumTrans<12><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 13) {
                kSupersampleMediumTrans<13><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 14) {
                kSupersampleMediumTrans<14><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 15) {
                kSupersampleMediumTrans<15><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 16) {
                kSupersampleMediumTrans<16><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            }
        } else {
            if(factor == 2) {
                kSupersampleMediumTransLoopy<2><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 3) {
                kSupersampleMediumTransLoopy<3><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 4) {
                kSupersampleMediumTransLoopy<4><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 5) {
                kSupersampleMediumTransLoopy<5><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 6) {
                kSupersampleMediumTransLoopy<6><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 7) {
                kSupersampleMediumTransLoopy<7><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 8) {
                kSupersampleMediumTransLoopy<8><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 9) {
                kSupersampleMediumTransLoopy<9><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 10) {
                kSupersampleMediumTransLoopy<10><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 11) {
                kSupersampleMediumTransLoopy<11><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 12) {
                kSupersampleMediumTransLoopy<12><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 13) {
                kSupersampleMediumTransLoopy<13><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 14) {
                kSupersampleMediumTransLoopy<14><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 15) {
                kSupersampleMediumTransLoopy<15><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            } else if(factor == 16) {
                kSupersampleMediumTransLoopy<16><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), numImages*imgSize, imgSize, shmemX);
            }
        }
    }
    cutilCheckMsg("kernel execution failed");
}

void _gtm(NVMatrix* images, NVMatrix* targets, int squareSize, bool avoidBankConflicts, bool reverse) {
    assert(!images->isTrans());
    int imgPixels = reverse ? targets->getNumCols() : images->getNumCols();
    int numImages = reverse ? targets->getNumRows() : images->getNumRows();
//    printf("images: %dx%d\n", images->getNumRows(), images->getNumCols());
//    printf("targets: %dx%d\n", targets->getNumRows(), targets->getNumCols());
//    printf("imgPixels: %d\n", imgPixels);
    assert(sqrt(float(imgPixels)) == floor(sqrt(float(imgPixels))));
    int imgSize = sqrt(imgPixels);
    assert(squareSize > 1 && squareSize <= 16);
    assert(imgSize > 0 && imgSize <= 512);
//    assert(squareSize * imgSize <= 512);
    assert(imgSize % squareSize == 0);
    assert(imgSize > squareSize);
    assert(targets->getNumElements() == images->getNumElements());

    bool useLoopy = false;
    int SHMEM_MAX = 8192; // don't use more than this much shmem
    int THREADS_MAX = 512;

    int threadsX = imgSize;
    int threadsY = squareSize * MIN(THREADS_MAX / (squareSize*threadsX), SHMEM_MAX / (4*threadsX*squareSize)); // to avoid running out of shmem
    if (threadsY == 0) {
        threadsX = 16;
        threadsY = squareSize * MIN(THREADS_MAX / (squareSize*threadsX), SHMEM_MAX / (4*threadsX*squareSize));
        useLoopy = true;
//        printf("using loopy\n");
    }

    int shmemX = squareSize;
    int shmemPadX = avoidBankConflicts * (1 - (shmemX % 2));
    shmemX += shmemPadX;
    int shmemY = threadsX * (threadsY / squareSize);

    int loopsYPerBlock = useLoopy ? GTM_LOOPY_BLOCK_LOOPS_Y : GTM_BLOCK_LOOPS_Y;
    int blocksX = imgSize;
    int blocksY = DIVUP(numImages, loopsYPerBlock * threadsY);
//    printf("boundary problems: %u\n", numImages % (loopsYPerBlock*threadsY) != 0);

    int shmem = 4 * shmemX * shmemY;
    if (shmem == 0 || shmem > 16300) {
        // this really shouldn't happen and i've only put this here as a precautionary measure
        // to avoid getting mysteriously wrong results.
        fprintf(stderr, "_gtm: not enough shared memory!");
        exit(EXIT_FAILURE);
    }

    dim3 grid(blocksX, blocksY);
    dim3 threads(threadsX, threadsY);
//    printf("blocks: %dx%d, threads: %dx%d\n", blocksY, blocksX, threadsY, threadsX);
//    printf("using %dx%d = %d bytes of shmem\n", shmemY, shmemX, shmem);

    if(reverse) {
        if(!useLoopy) {
            if(squareSize == 2) {
                kGridToMatrix<2, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 3) {
                kGridToMatrix<3, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 4) {
                kGridToMatrix<4, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 5) {
                kGridToMatrix<5, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 6) {
                kGridToMatrix<6, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 7) {
                kGridToMatrix<7, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 8) {
                kGridToMatrix<8, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 9) {
                kGridToMatrix<9, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 10) {
                kGridToMatrix<10, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 11) {
                kGridToMatrix<11, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 12) {
                kGridToMatrix<12, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 13) {
                kGridToMatrix<13, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 14) {
                kGridToMatrix<14, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 15) {
                kGridToMatrix<15, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 16) {
                kGridToMatrix<16, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            }
        } else {
            if(squareSize == 2) {
                kGridToMatrixLoopy<2, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 3) {
                kGridToMatrixLoopy<3, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 4) {
                kGridToMatrixLoopy<4, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 5) {
                kGridToMatrixLoopy<5, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 6) {
                kGridToMatrixLoopy<6, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 7) {
                kGridToMatrixLoopy<7, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 8) {
                kGridToMatrixLoopy<8, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 9) {
                kGridToMatrixLoopy<9, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 10) {
                kGridToMatrixLoopy<10, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 11) {
                kGridToMatrixLoopy<11, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 12) {
                kGridToMatrixLoopy<12, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 13) {
                kGridToMatrixLoopy<13, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 14) {
                kGridToMatrixLoopy<14, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 15) {
                kGridToMatrixLoopy<15, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 16) {
                kGridToMatrixLoopy<16, true><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            }
        }
    } else {
        if(!useLoopy) {
            if(squareSize == 2) {
                kGridToMatrix<2, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 3) {
                kGridToMatrix<3, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 4) {
                kGridToMatrix<4, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 5) {
                kGridToMatrix<5, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 6) {
                kGridToMatrix<6, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 7) {
                kGridToMatrix<7, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 8) {
                kGridToMatrix<8, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 9) {
                kGridToMatrix<9, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 10) {
                kGridToMatrix<10, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 11) {
                kGridToMatrix<11, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 12) {
                kGridToMatrix<12, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 13) {
                kGridToMatrix<13, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 14) {
                kGridToMatrix<14, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 15) {
                kGridToMatrix<15, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 16) {
                kGridToMatrix<16, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            }
        } else {
            if(squareSize == 2) {
                kGridToMatrixLoopy<2, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 3) {
                kGridToMatrixLoopy<3, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 4) {
                kGridToMatrixLoopy<4, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 5) {
                kGridToMatrixLoopy<5, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 6) {
                kGridToMatrixLoopy<6, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 7) {
                kGridToMatrixLoopy<7, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 8) {
                kGridToMatrixLoopy<8, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 9) {
                kGridToMatrixLoopy<9, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 10) {
                kGridToMatrixLoopy<10, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 11) {
                kGridToMatrixLoopy<11, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 12) {
                kGridToMatrixLoopy<12, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 13) {
                kGridToMatrixLoopy<13, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 14) {
                kGridToMatrixLoopy<14, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 15) {
                kGridToMatrixLoopy<15, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            } else if(squareSize == 16) {
                kGridToMatrixLoopy<16, false><<<grid, threads, shmem>>>(images->getDevData(), targets->getDevData(), imgSize, numImages*imgSize, shmemX);
            }
        }
    }
    cutilCheckMsg("kernel execution failed");
}

void gridToMatrix(NVMatrix* images, NVMatrix* targets, int squareSize, bool avoidBankConflicts) {
    _gtm(images, targets, squareSize, avoidBankConflicts, false);
}


void matrixToGrid(NVMatrix* images, NVMatrix* targets, int squareSize, bool avoidBankConflicts) {
    _gtm(images, targets, squareSize, avoidBankConflicts, true);
}

/*
 * Samples from a bunch of multinomial distributions, where each row of the "multi" matrix
 * is a different distribution. Of course, each row of the "multi" matrix must sum to 1.
 *
 * It's optimized for the case when you want to sample from lots (hundreds of thousands)
 * of fairly small multinomial distributions.
 *
 * The case when the multinomials are in columns is much easier and faster.
 */
void sampleMultinomial(NVMatrix* multi, NVMatrix* randoms, NVMatrix* targets) {
    assert(!multi->isTrans());
    assert(multi->isSameDims(*targets));
    assert(multi->getNumCols() <= 1024);
    assert(randoms->getNumElements() == multi->getNumRows());
    int nomials = multi->getNumCols();
    int multinomials = multi->getNumRows();

    if(nomials > 256 || multinomials < 8192) {
        /*
         * I'm really not sure about the merits of this tree-based function. I may
         * remove it in the future. It's faster in some cases (e.g. when the number of
         * multinomials is small and the multinomials are very large), but you can get
         * similar performance from the non-tree-based one by reducing the number of
         * y-loops.
         */
        dim3 grid(1, DIVUP(multinomials, 1));
        while (grid.y > NUM_BLOCKS_MAX) {
            grid.y = DIVUP(grid.y, 2);
            grid.x *= 2;
        }
    //    printf("grid: %dx%d\n", grid.x, grid.y);
        if(nomials <= 64) { // yes i know this can't happen under current conditions
            dim3 threads(32, 1);
            kSampleMultinomial<32><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(), nomials, multinomials);
        } else if(nomials <= 128) {
            dim3 threads(64, 1);
            kSampleMultinomial<64><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(), nomials, multinomials);
        } else if(nomials <= 256) {
            dim3 threads(128, 1);
            kSampleMultinomial<128><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(), nomials, multinomials);
        } else if(nomials <= 512) {
            dim3 threads(256, 1);
            kSampleMultinomial<256><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(), nomials, multinomials);
        } else {
            dim3 threads(512, 1);
            kSampleMultinomial<512><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(), nomials, multinomials);
        }
    } else {
        dim3 grid(1,DIVUP(multinomials, SSM_THREADS_Y*SSM_LOOPS_Y));
        dim3 threads(SSM_THREADS_X, SSM_THREADS_Y);

        while (grid.y > NUM_BLOCKS_MAX) {
            grid.y = DIVUP(grid.y, 2);
            grid.x *= 2;
        }
        if(nomials <= 16) {
            if(nomials <= 4) {
                kSampleSmallMultinomial<1, 4><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
            } else if(nomials <= 8) {
                kSampleSmallMultinomial<1, 8><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
            } else if(nomials <= 12) {
                kSampleSmallMultinomial<1, 12><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
            } else {
                kSampleSmallMultinomial<1, 16><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
            }
        } else if(nomials <= 32) {
            kSampleSmallMultinomial<2, SSM_THREADS_X><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
        } else if(nomials <= 48){
            kSampleSmallMultinomial<3, SSM_THREADS_X><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
        } else if(nomials <= 64){
            kSampleSmallMultinomial<4, SSM_THREADS_X><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
        } else if(nomials <= 80){
            kSampleSmallMultinomial<5, SSM_THREADS_X><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
        } else if(nomials <= 96){
            kSampleSmallMultinomial<6, SSM_THREADS_X><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
        } else if(nomials <= 112){
            kSampleSmallMultinomial<7, SSM_THREADS_X><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
        } else if(nomials <= 128){
            kSampleSmallMultinomial<8, SSM_THREADS_X><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
        } else if(nomials <= 144){
            kSampleSmallMultinomial<9, SSM_THREADS_X><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
        } else if(nomials <= 160){
            kSampleSmallMultinomial<10, SSM_THREADS_X><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
        } else if(nomials <= 176){
            kSampleSmallMultinomial<11, SSM_THREADS_X><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
        } else if(nomials <= 192){
            kSampleSmallMultinomial<12, SSM_THREADS_X><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
        } else if(nomials <= 208){
            kSampleSmallMultinomial<13, SSM_THREADS_X><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
        } else if(nomials <= 224){
            kSampleSmallMultinomial<14, SSM_THREADS_X><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
        } else if(nomials <= 240){
            kSampleSmallMultinomial<15, SSM_THREADS_X><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
        } else if(nomials <= 256){
            kSampleSmallMultinomial<16, SSM_THREADS_X><<<grid, threads>>>(multi->getDevData(), randoms->getDevData(), targets->getDevData(),nomials, multinomials);
        }
    }
    cutilCheckMsg("kernel execution failed");
}
