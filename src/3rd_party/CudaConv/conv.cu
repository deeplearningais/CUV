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
 * conv.cu
 *
 *  Created on: Oct 31, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#include <math.h>
#include <nvmatrix.cuh>
#include "conv.cuh"

void _convolve_bw(float* images, float* filters, float* targets, int numImgsPerGroup,
                  int numFiltersPerGroup, int numGroups, int imgSize, int filterSize, int stride, bool useDynamics = false) {
    assert(stride == 1 || stride == 3);
    int numOutputsX = imgSize - filterSize + 1;
//    int numOutputs = numOutputsX*numOutputsX;
    bool checkOutputBounds = numOutputsX % 16 != 0;
    if(filterSize > 20) {
        bool checkFilterBounds = filterSize % 16 != 0;
        int threadsZ = numFiltersPerGroup > 8 ? 8 : numFiltersPerGroup > 4 ? 4 : 2;
        int blocksY = DIVUP(numFiltersPerGroup, 2*threadsZ), blocksX = numImgsPerGroup * numGroups;
        dim3 grid(blocksX, blocksY);
        dim3 threads(16, 4, threadsZ);

        if(threadsZ == 8) {
            if(checkFilterBounds) {
                if(stride == 1) {
                    conv_bw_nofit_4x16_2per<true, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_4x16_2per<true, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            } else {
                if(stride == 1) {
                    conv_bw_nofit_4x16_2per<false, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_4x16_2per<false, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if(threadsZ == 4) {
            if(checkFilterBounds) {
                if(stride == 1) {
                    conv_bw_nofit_4x16_2per<true, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_4x16_2per<true, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            } else {
                if(stride == 1) {
                    conv_bw_nofit_4x16_2per<false, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_4x16_2per<false, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if(threadsZ == 2) {
            if(checkFilterBounds) {
                if(stride == 1) {
                    conv_bw_nofit_4x16_2per<true, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_4x16_2per<true, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                }
            } else {
                if(stride == 1) {
                    conv_bw_nofit_4x16_2per<false, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_nofit_4x16_2per<false, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFiltersPerGroup, numGroups);
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
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<15, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<15, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 4){
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<15, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<15, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 2){
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<15, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<15, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if(filterSize == 16) {
            if(threadsZ == 8) {
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<16, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<16, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 4){
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<16, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<16, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 2){
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<16, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<16, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if(filterSize == 17) {
            if(threadsZ == 8) {
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<17, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<17, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 4){
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<17, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<17, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 2){
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<17, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<17, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if(filterSize == 18) {
            if(threadsZ == 8) {
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<18, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<18, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 4){
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<18, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<18, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 2){
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<18, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<18, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if(filterSize == 19) {
            if(threadsZ == 8) {
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<19, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<19, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 4){
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<19, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<19, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 2){
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<19, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<19, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        } else if(filterSize == 20) {
            if(threadsZ == 8) {
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<20, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<20, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 4){
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<20, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<20, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            } else if(threadsZ == 2){
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<20, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                } else {
                    conv_bw_fit_4x16_1per<20, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                }
            }
        }
    } else {
        if (useDynamics) {
            // later
        } else {
            int threadsZ = numFiltersPerGroup > 8 ? 8 : numFiltersPerGroup > 4 ? 4 : 2;
            int blocksY = DIVUP(numFiltersPerGroup, 2*threadsZ), blocksX = numImgsPerGroup * numGroups;
            dim3 grid(blocksX, blocksY);
            dim3 threads(16, 4, threadsZ);
//            printf("numFiltersPerGroup: %d, numImgsPerGroup: %d, numGroups: %d\n", numFiltersPerGroup, numImgsPerGroup, numGroups);
            if (filterSize == 1) {
                throw "try multByScalar";
            } else if (filterSize == 2) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<2, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<2, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<2, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<2, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<2, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<2, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (filterSize == 3) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<3, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<3, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<3, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<3, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<3, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<3, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 4) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<4, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<4, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<4, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<4, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<4, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<4, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 5) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<5, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<5, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<5, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<5, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<5, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<5, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 6) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<6, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<6, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<6, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<6, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<6, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<6, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 7) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<7, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<7, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<7, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<7, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<7, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<7, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 8) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<8, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<8, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<8, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<8, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<8, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<8, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 9) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<9, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<9, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<9, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<9, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<9, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<9, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 10) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<10, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<10, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<10, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<10, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<10, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<10, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 11) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<11, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<11, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<11, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<11, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<11, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<11, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 12) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<12, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<12, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<12, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<12, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<12, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<12, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }  else if (filterSize == 13) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<13, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<13, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<13, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<13, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<13, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<13, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            } else if (filterSize == 14) {
                if (threadsZ == 8) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<14, 1, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<14, 3, 8, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);

                    }
                } else if (threadsZ == 4) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<14, 1, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<14, 3, 4, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                } else if (threadsZ == 2) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<14, 1, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    } else {
                        conv_bw_fit_4x16_2per<14, 3, 2, false><<<grid, threads>>>(images, filters, targets, imgSize, numFiltersPerGroup, numGroups);
                    }
                }
            }
        }
    }
    cutilCheckMsg("kernel execution failed");
}

void convolve(NVMatrix* images, NVMatrix* filters, NVMatrix* targets, int numGroups, bool color) {
    int colorMult = color ? 3 : 1;
    assert(images->getNumCols() % colorMult == 0);
    assert(filters->getNumCols() % colorMult == 0);
    double dImgSize = sqrt(images->getNumCols() / colorMult);
    double dFilterSize = sqrt(filters->getNumCols() / colorMult);
    assert(dImgSize == floor(dImgSize));
    assert(dFilterSize == floor(dFilterSize));
    assert(images->getNumRows() % numGroups == 0);
    assert(filters->getNumRows() % numGroups == 0);
    int imgSize = int(dImgSize);
    int filterSize = int(dFilterSize);
    int numImgsPerGroup = images->getNumRows() / numGroups;
    int numFiltersPerGroup = filters->getNumRows() / numGroups;
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;

    assert(numFiltersPerGroup % 2 == 0);
    assert(targets->getNumElements() == numOutputs * numFiltersPerGroup * numImgsPerGroup * numGroups);
    assert(!images->isTrans());
    assert(!filters->isTrans());
    assert(!targets->isTrans());
    assert(imgSize > filterSize);

    if(!color) {
        _convolve_bw(images->getDevData(), filters->getDevData(), targets->getDevData(),
                     numImgsPerGroup, numFiltersPerGroup, numGroups, imgSize, filterSize, 1);
    } else {
        targets->apply(NVMatrix::ZERO);
        _convolve_bw(images->getDevData(), filters->getDevData(), targets->getDevData(),
                     numImgsPerGroup, numFiltersPerGroup, numGroups, imgSize, filterSize, 3);
        _convolve_bw(images->getDevData() + imgPixels, filters->getDevData() + filterPixels, targets->getDevData(),
                     numImgsPerGroup, numFiltersPerGroup, numGroups, imgSize, filterSize, 3);
        _convolve_bw(images->getDevData() + 2*imgPixels, filters->getDevData() + 2*filterPixels, targets->getDevData(),
                     numImgsPerGroup, numFiltersPerGroup, numGroups, imgSize, filterSize, 3);
    }
}
