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
 * conv2.cu
 *
 *  Created on: Nov 10, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */


#include <math.h>
#include <3rd_party/CudaConv/nvmatrix.cuh>
#include "conv.cuh"
#include <cuv/tools/cuv_general.hpp>

void _convolve2_bw(float* images, float* filters, float* targets, int numImgsPerGroup,
                  int numFiltersPerGroup, int imgSize, int filterSize, int imagesPerFilter, int numGroups, bool useDynamics = false) {
    assert(imagesPerFilter == 1 || imagesPerFilter == 3);
    int numOutputsX = imgSize - filterSize + 1;
    /*bool checkOutputBounds = numOutputsX % 16 != 0;*/
    if (numOutputsX <= 9) {
        /*
         * Call special dynamic routine which is fast when the number of outputs is small.
         */
        int threadsX = numOutputsX, threadsY = numOutputsX, threadsZ = 512 / (threadsX*threadsY);
        int blocksX = numImgsPerGroup * numGroups, blocksY = DIVUP(numFiltersPerGroup, threadsZ*2);
        bool checkFilterBounds = filterSize % threadsX != 0;
//        bool checkFilterIdxBounds = numFiltersPerGroup % (threadsZ*2) != 0;

        dim3 grid(blocksX, blocksY);
        dim3 threads(threadsX, threadsY, threadsZ);

        if (threadsX == 2) {
            if (checkFilterBounds) {
                if (imagesPerFilter == 1) {
                    conv_bw_nofit_dynXYZ_2per<true, 1, 2, 128, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_dynXYZ_2per<true, 3, 2, 128, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            } else {
                if (imagesPerFilter == 1) {
                    conv_bw_nofit_dynXYZ_2per<false, 1, 2, 128, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_dynXYZ_2per<false, 3, 2, 128, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if (threadsX == 3) {
            if (checkFilterBounds) {
                if (imagesPerFilter == 1) {
                    conv_bw_nofit_dynXYZ_2per<true, 1, 3, 56, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_dynXYZ_2per<true, 3, 3, 56, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            } else {
                if (imagesPerFilter == 1) {
                    conv_bw_nofit_dynXYZ_2per<false, 1, 3, 56, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_dynXYZ_2per<false, 3, 3, 56, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if (threadsX == 4) {
            if (checkFilterBounds) {
                if (imagesPerFilter == 1) {
                    conv_bw_nofit_dynXYZ_2per<true, 1, 4, 32, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_dynXYZ_2per<true, 3, 4, 32, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            } else {
                if (imagesPerFilter == 1) {
                    conv_bw_nofit_dynXYZ_2per<false, 1, 4, 32, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_dynXYZ_2per<false, 3, 4, 32, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if (threadsX == 5) {
            if (checkFilterBounds) {
                if (imagesPerFilter == 1) {
                    conv_bw_nofit_dynXYZ_2per<true, 1, 5, 20, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_dynXYZ_2per<true, 3, 5, 20, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            } else {
                if (imagesPerFilter == 1) {
                    conv_bw_nofit_dynXYZ_2per<false, 1, 5, 20, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_dynXYZ_2per<false, 3, 5, 20, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if (threadsX == 6) {
            if (checkFilterBounds) {
                if (imagesPerFilter == 1) {
                    conv_bw_nofit_dynXYZ_2per<true, 1, 6, 14, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_dynXYZ_2per<true, 3, 6, 14, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            } else {
                if (imagesPerFilter == 1) {
                    conv_bw_nofit_dynXYZ_2per<false, 1, 6, 14, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_dynXYZ_2per<false, 3, 6, 14, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if (threadsX == 7) {
            if (checkFilterBounds) {
                if (imagesPerFilter == 1) {
                    conv_bw_nofit_dynXYZ_2per<true, 1, 7, 10, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_dynXYZ_2per<true, 3, 7, 10, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            } else {
                if (imagesPerFilter == 1) {
                    conv_bw_nofit_dynXYZ_2per<false, 1, 7, 10, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_dynXYZ_2per<false, 3, 7, 10, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if (threadsX == 8) {
            if (checkFilterBounds) {
                if (imagesPerFilter == 1) {
                    conv_bw_nofit_dynXYZ_2per<true, 1, 8, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_dynXYZ_2per<true, 3, 8, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            } else {
                if (imagesPerFilter == 1) {
                    conv_bw_nofit_dynXYZ_2per<false, 1, 8, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_dynXYZ_2per<false, 3, 8, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if (threadsX == 9) {
            if (checkFilterBounds) {
                if (imagesPerFilter == 1) {
                    conv_bw_nofit_dynXYZ_2per<true, 1, 9, 6, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_dynXYZ_2per<true, 3, 9, 6, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            } else {
                if (imagesPerFilter == 1) {
                    conv_bw_nofit_dynXYZ_2per<false, 1, 9, 6, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_dynXYZ_2per<false, 3, 9, 6, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            }
        }
    } else if(filterSize > 20) {
        bool checkFilterBounds = filterSize % 16 != 0;
        int threadsZ = numFiltersPerGroup > 8 ? 8 : numFiltersPerGroup > 4 ? 4 : 2;
        int blocksY = DIVUP(numFiltersPerGroup, 2*threadsZ), blocksX = numImgsPerGroup * numGroups;
        dim3 grid(blocksX, blocksY);
        dim3 threads(16, 4, threadsZ);

        if(threadsZ == 8) {
            if(checkFilterBounds) {
                if(imagesPerFilter == 1) {
                    conv_bw_nofit_4x16_2per<true, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_4x16_2per<true, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            } else {
                if(imagesPerFilter == 1) {
                    conv_bw_nofit_4x16_2per<false, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_4x16_2per<false, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if(threadsZ == 4) {
            if(checkFilterBounds) {
                if(imagesPerFilter == 1) {
                    conv_bw_nofit_4x16_2per<true, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_4x16_2per<true, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            } else {
                if(imagesPerFilter == 1) {
                    conv_bw_nofit_4x16_2per<false, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_4x16_2per<false, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if(threadsZ == 2) {
            if(checkFilterBounds) {
                if(imagesPerFilter == 1) {
                    conv_bw_nofit_4x16_2per<true, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_4x16_2per<true, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            } else {
                if(imagesPerFilter == 1) {
                    conv_bw_nofit_4x16_2per<false, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_4x16_2per<false, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            }
        }
    } else if (filterSize > 14) {
        int threadsZ = numFiltersPerGroup >= 8 ? 8 : numFiltersPerGroup >= 4 ? 4 : 2;
        int blocksY = DIVUP(numFiltersPerGroup, threadsZ), blocksX = numImgsPerGroup * numGroups;
        dim3 grid(blocksX, blocksY);
        dim3 threads(16, 4, threadsZ);
        if(filterSize == 15) {
            if(threadsZ == 8) {
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<15, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<15, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 4){
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<15, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<15, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 2){
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<15, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<15, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if(filterSize == 16) {
            if(threadsZ == 8) {
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<16, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<16, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 4){
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<16, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<16, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 2){
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<16, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<16, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if(filterSize == 17) {
            if(threadsZ == 8) {
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<17, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<17, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 4){
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<17, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<17, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 2){
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<17, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<17, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if(filterSize == 18) {
            if(threadsZ == 8) {
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<18, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<18, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 4){
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<18, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<18, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 2){
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<18, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<18, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if(filterSize == 19) {
            if(threadsZ == 8) {
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<19, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<19, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 4){
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<19, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<19, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 2){
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<19, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<19, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if(filterSize == 20) {
            if(threadsZ == 8) {
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<20, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<20, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 4){
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<20, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<20, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 2){
                if(imagesPerFilter == 1) {
                    conv_bw_fit_4x16_1per<20, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<20, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        }
    } else {
        int threadsZ = numFiltersPerGroup > 8 ? 8 : numFiltersPerGroup > 4 ? 4 : 2;
        int blocksY = DIVUP(numFiltersPerGroup, 2*threadsZ), blocksX = numImgsPerGroup * numGroups;
        dim3 grid(blocksX, blocksY);
        dim3 threads(16, 4, threadsZ);
//        printf("calling unified conv1/2 routine\n");
//            printf("blocks x: %d, blocks y: %d\n", blocksX, blocksY);
        if (filterSize == 1) {
            throw "try multByScalar";
        } else if (filterSize == 2) {
            if (threadsZ == 8) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<2, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<2, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                }
            } else if (threadsZ == 4) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<2, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<2, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if (threadsZ == 2) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<2, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<2, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if (filterSize == 3) {
            if (threadsZ == 8) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<3, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<3, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                }
            } else if (threadsZ == 4) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<3, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<3, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if (threadsZ == 2) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<3, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<3, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        }  else if (filterSize == 4) {
            if (threadsZ == 8) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<4, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<4, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                }
            } else if (threadsZ == 4) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<4, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<4, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if (threadsZ == 2) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<4, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<4, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        }  else if (filterSize == 5) {
            if (threadsZ == 8) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<5, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<5, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                }
            } else if (threadsZ == 4) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<5, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<5, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if (threadsZ == 2) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<5, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<5, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        }  else if (filterSize == 6) {
            if (threadsZ == 8) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<6, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<6, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                }
            } else if (threadsZ == 4) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<6, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<6, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if (threadsZ == 2) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<6, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<6, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        }  else if (filterSize == 7) {
            if (threadsZ == 8) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<7, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<7, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                }
            } else if (threadsZ == 4) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<7, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<7, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if (threadsZ == 2) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<7, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<7, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        }  else if (filterSize == 8) {
            if (threadsZ == 8) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<8, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<8, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                }
            } else if (threadsZ == 4) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<8, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<8, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if (threadsZ == 2) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<8, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<8, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        }  else if (filterSize == 9) {
            if (threadsZ == 8) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<9, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<9, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                }
            } else if (threadsZ == 4) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<9, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<9, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if (threadsZ == 2) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<9, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<9, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        }  else if (filterSize == 10) {
            if (threadsZ == 8) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<10, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<10, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                }
            } else if (threadsZ == 4) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<10, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<10, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if (threadsZ == 2) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<10, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<10, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        }  else if (filterSize == 11) {
            if (threadsZ == 8) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<11, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<11, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                }
            } else if (threadsZ == 4) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<11, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<11, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if (threadsZ == 2) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<11, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<11, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        }  else if (filterSize == 12) {
            if (threadsZ == 8) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<12, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<12, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                }
            } else if (threadsZ == 4) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<12, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<12, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if (threadsZ == 2) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<12, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<12, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        }  else if (filterSize == 13) {
            if (threadsZ == 8) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<13, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<13, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                }
            } else if (threadsZ == 4) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<13, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<13, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if (threadsZ == 2) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<13, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<13, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if (filterSize == 14) {
            if (threadsZ == 8) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<14, 1, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<14, 3, 8, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                }
            } else if (threadsZ == 4) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<14, 1, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<14, 3, 4, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if (threadsZ == 2) {
                if (imagesPerFilter == 1) {
                    conv_bw_fit_4x16_2per<14, 1, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_2per<14, 3, 2, true><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        }
    }
	cuvSafeCall(cudaThreadSynchronize());
    /*cutilCheckMsg("kernel execution failed");*/
}
/*
 * Here the "filters" might represent the activities of the hidden layer of a convolutional net
 * (the output of a convolution), so the color attribute does not apply to them.
 */
void convolve2(NVMatrix* images, NVMatrix* filters, NVMatrix* targets, int filterSize, int numGroups, bool colorImages) {
    int colorMult = colorImages ? 3 : 1;
    assert(images->getNumCols() % colorMult == 0);
    double dImgSize = sqrt(images->getNumCols() / colorMult);
    assert(dImgSize == floor(dImgSize));
    // each row in "filters" corresponds to a set of filters to convolve with the images in the same row of "images"
//    assert(images->getNumRows() == filters->getNumRows());
//    assert(filters->getNumCols() % filterPixels == 0);
    assert(images->getNumRows() % numGroups == 0);
    assert(filters->getNumRows() % numGroups == 0);
    //    assert(dFilterSize == floor(dFilterSize));
    int imgSize = int(dImgSize);
    //    int filterSize = int(dFilterSize);
    int numImgsPerGroup = images->getNumRows() / numGroups;
    int numFiltersPerGroup = filters->getNumRows() / numGroups;
    /*int numOutputsX = (imgSize - filterSize + 1);*/

    assert(filters->getNumCols() == numImgsPerGroup * filterSize*filterSize);
    assert(numFiltersPerGroup % 2 == 0);
    assert(targets->getNumElements() == (imgSize - filterSize + 1)*(imgSize - filterSize + 1) * numFiltersPerGroup * numImgsPerGroup * numGroups * colorMult);
    assert(!images->isTrans());
    assert(!filters->isTrans());
    assert(!targets->isTrans());
    assert(imgSize > filterSize);

    _convolve2_bw(images->getDevData(), filters->getDevData(), targets->getDevData(),
                 numImgsPerGroup * colorMult, numFiltersPerGroup, imgSize, filterSize, colorMult, numGroups);
}
