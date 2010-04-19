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
 * convCPU.cpp
 *
 *  Created on: Oct 31, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#include "convCPU.h"

inline float dotCPU(float* img, float* filter, int imgSize, int filterSize, int y, int x) {
    float result = 0;
    for(int fY = 0; fY < filterSize; fY++) {
        for(int fX = 0; fX < filterSize; fX++) {
            result += img[(y+fY) * imgSize + fX + x] * filter[fY * filterSize + fX];
        }
    }
    return result;
}

void convCPU(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgs, int numFilters) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for(int i = 0; i < numImgs; i++) {
        for(int f = 0; f < numFilters; f++) {
            for(int y = 0; y < numOutputsX; y++) {
                for(int x = 0; x < numOutputsX; x++) {
                    targets[f * numOutputs + y * numOutputsX + x] = dotCPU(imgs, &filters[f*filterPixels], imgSize, filterSize, y, x);
                }
            }
        }
        imgs += imgPixels;
        targets += numOutputs * numFilters;
    }
}

void convColorCPU(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgs, int numFilters) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for(int i = 0; i < numImgs; i++) {
        for(int f = 0; f < numFilters; f++) {
            for(int y = 0; y < numOutputsX; y++) {
                for(int x = 0; x < numOutputsX; x++) {
                    targets[f * numOutputs + y * numOutputsX + x] = dotCPU(imgs, &filters[f*filterPixels*3], imgSize, filterSize, y, x)
                                                                    + dotCPU(imgs + imgPixels, &filters[f*filterPixels*3 + filterPixels], imgSize, filterSize, y, x)
                                                                    + dotCPU(imgs + 2*imgPixels, &filters[f*filterPixels*3 + 2*filterPixels], imgSize, filterSize, y, x);
                }
            }
        }
        imgs += imgPixels*3;
        targets += numOutputs * numFilters;
    }
}

void conv2CPU(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgs, int numFilters) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for(int i = 0; i < numImgs; i++) {
        for(int f = 0; f < numFilters; f++) {
            for(int y = 0; y < numOutputsX; y++) {
                for(int x = 0; x < numOutputsX; x++) {
                    targets[f * numOutputs + y * numOutputsX + x] = dotCPU(imgs, &filters[i * numFilters * filterPixels + f*filterPixels], imgSize, filterSize, y, x);
                }
            }
        }
        imgs += imgPixels;
        targets += numOutputs * numFilters;
    }
}

void conv2ColorCPU(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgs, int numFilters) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for(int i = 0; i < numImgs; i++) {
        for(int f = 0; f < numFilters; f++) {
            for(int y = 0; y < numOutputsX; y++) {
                for(int x = 0; x < numOutputsX; x++) {
                    targets[f * numOutputs + y * numOutputsX + x] = dotCPU(imgs, &filters[(i/3) * numFilters * filterPixels + f*filterPixels], imgSize, filterSize, y, x);
                }
            }
        }
        imgs += imgPixels;
        targets += numOutputs * numFilters;
    }
}

void rotate180CPU(float* filters, float* targets, int filterSize, int numFilters) {
    int filterPixels = filterSize * filterSize;
    for(int f = 0; f < numFilters; f++) {
        for(int y = 0; y < filterSize; y++) {
            for(int x = 0; x < filterSize; x++) {
                targets[f * filterPixels + (filterSize  - 1 - y) * filterSize + filterSize -1 - x] = filters[f * filterPixels + y * filterSize + x];
            }
        }
    }
}

void padZerosCPU(float* images, float* targets, int imgSize, int numImages, int paddingSize) {
    int targetSize = imgSize + 2*paddingSize;
    for(int i = 0; i < numImages; i++) {
        targets += paddingSize * targetSize;
        for(int y = 0; y < imgSize; y++) {
            targets += paddingSize;
            for(int x = 0; x < imgSize; x++) {
                targets[0] = images[0];

                images++;
                targets++;
            }
            targets += paddingSize;
        }
        targets += paddingSize * targetSize;
    }
}

inline float dotRotateCPU(float* img, float* filter, int imgSize, int filterSize, int y, int x) {
    img += y * imgSize;
    float result = 0;
    for (int fY = 0; fY < filterSize; fY++) {
        for (int fX = 0; fX < filterSize; fX++) {
            result += img[fX + x] * filter[(filterSize - 1 - fY) * filterSize + filterSize - 1 - fX];
        }
        img += imgSize;
    }
    return result;
}

void conv3CPU(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgs, int numFilters) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for (int i = 0; i < numImgs; i++) {
        for (int f = 0; f < numFilters; f++) {
            for (int y = 0; y < numOutputsX; y++) {
                for (int x = 0; x < numOutputsX; x++) {
                    targets[y * numOutputsX + x] += dotRotateCPU(&imgs[imgPixels * f], &filters[f * filterPixels], imgSize, filterSize, y, x);
                }
            }
//            return;
        }
        imgs += imgPixels * numFilters;
        targets += numOutputs;
    }
}

void conv3ColorCPU(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgs, int numFilters) {
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;
    for (int i = 0; i < numImgs; i++) {
        for (int f = 0; f < numFilters; f++) {
            for (int y = 0; y < numOutputsX; y++) {
                for (int x = 0; x < numOutputsX; x++) {
                    targets[y * numOutputsX + x + (f % 3) * numOutputs] += dotRotateCPU(&imgs[imgPixels * (f / 3)], &filters[f * filterPixels], imgSize, filterSize, y, x);
                }
            }
        }
        imgs += imgPixels * (numFilters / 3);
        targets += 3 * numOutputs;
    }
}

inline float sumCPU(float* image, int imgSize, int factor) {
    float sum = 0;
    for(int y = 0; y < factor; y++) {
        for(int x = 0; x < factor; x++) {
            sum += image[imgSize * y + x];
        }
    }
    return sum;
}

void subsampleCPU(float* images, float* targets, int imgSize, int factor, int numImgs) {
    int numRegions = imgSize / factor;
    float divisor = factor * factor;
    for(int i = 0; i < numImgs; i++) {
        for(int y = 0; y < numRegions; y++) {
            for(int x = 0; x < numRegions; x++) {
                targets[0] = sumCPU(&images[y * imgSize * factor + x * factor], imgSize, factor) / divisor;
                targets++;
            }
        }
        images += imgSize * imgSize;
    }
}

void supersampleCPU(float* images, float* targets, int imgSize, int factor, int numImgs, bool trans) {
    int targetSize = imgSize * factor;
    if (!trans) {
        for (int i = 0; i < numImgs; i++) {
            for (int y = 0; y < targetSize; y++) {
                for (int x = 0; x < targetSize; x++) {
                    targets[0] = images[(y / factor) * imgSize + x / factor];
                    targets++;
                }
            }
            images += imgSize * imgSize;
        }
    } else {
        for (int i = 0; i < numImgs; i++) {
            for (int y = 0; y < targetSize; y++) {
                for (int x = 0; x < targetSize; x++) {
                    targets[0] = images[(x / factor) * numImgs*imgSize + y / factor];
                    targets++;
                }
            }
            images += imgSize;
        }
    }
}

void gridToMatrixCPU(float* images, float* targets, int imgSize, int factor, int numImgs) {
    //    int targetSizeX = factor*factor;
    for (int i = 0; i < numImgs * (imgSize / factor); i++) {
        for (int y = 0; y < factor; y++) {
            for (int x = 0; x < imgSize; x++) {
                targets[x * factor + y] = images[0];
                images++;
            }
        }
        targets += factor * imgSize;
    }
}

void matrixToGridCPU(float* images, float* targets, int imgSize, int factor, int numImgs) {
    //    int targetSizeX = factor*factor;
    for (int i = 0; i < numImgs * (imgSize / factor); i++) {
        for (int y = 0; y < factor; y++) {
            for (int x = 0; x < imgSize; x++) {
                targets[0] = images[x * factor + y];
                targets++;
            }
        }
        images += factor * imgSize;
    }
}

void sampleMultinomialCPU(float* multi, float* randoms, float* targets, int multinomials, int nomials) {
    for(int i = 0; i < multinomials; i++) {
        float sum = 0, prevSum = 0;
        float rnd = randoms[i];
        for(int x = 0; x < nomials; x++) {
            sum += multi[0];
            targets[0] = rnd >= prevSum && rnd < sum;
//            targets[0] = sum;
//            sum += multi[0];
            targets++;
            multi++;
            prevSum = sum;
        }
    }
}
