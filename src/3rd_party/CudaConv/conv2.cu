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
#include <nvmatrix.cuh>
#include "conv2.cuh"

void _convolve2_bw(float* images, float* filters, float* targets, int numCases,
                  int numFilters, int imgSize, int filterSize, int imagesPerFilter, bool useDynamics = false) {
    assert(imagesPerFilter == 1 || imagesPerFilter == 3);
    int numOutputsX = imgSize - filterSize + 1;
//    int numOutputs = numOutputsX*numOutputsX;
    bool checkOutputBounds = numOutputsX % 16 != 0;
    if (/*false  &&*/numOutputsX <= 8) {
        /*
         * Call special dynamic routine which is fast when the number of outputs is small.
         */
        int threadsX = numOutputsX, threadsY = numOutputsX, threadsZ = 512 / (threadsX*threadsY);
        int blocksX = numCases, blocksY = DIVUP(numFilters, threadsZ*2);
        bool checkFilterBounds = filterSize % threadsX != 0;
        bool checkFilterIdxBounds = numFilters % (threadsZ*2) != 0;
        dim3 grid(blocksX, blocksY);
        dim3 threads(threadsX, threadsY, threadsZ);
//        printf("numcases: %d, numfilters: %d, imgsize: %d, filtersize: %d\n", numCases, numFilters, imgSize, filterSize);
//        printf("check filter bds: %d, idx bds: %d\n", checkFilterBounds, checkFilterIdxBounds);
//        printf("grid: %dx%d\n", grid.x, grid.y);
//        printf("threads: %dx%dx%d\n", threads.x, threads.y, threads.z);

        if (threadsX == 2) {
            if (checkFilterBounds) {
                if(checkFilterIdxBounds) {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<true, true, 1, 2, 128><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<true, true, 3, 2, 128><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                } else {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<true, false, 1, 2, 128><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<true, false, 3, 2, 128><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                }
            } else {
                if(checkFilterIdxBounds) {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<false, true, 1, 2, 128><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<false, true, 3, 2, 128><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                } else {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<false, false, 1, 2, 128><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<false, false, 3, 2, 128><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                }
            }
        } else if (threadsX == 3) {
            if (checkFilterBounds) {
                if(checkFilterIdxBounds) {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<true, true, 1, 3, 56><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<true, true, 3, 3, 56><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                } else {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<true, false, 1, 3, 56><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<true, false, 3, 3, 56><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                }
            } else {
                if(checkFilterIdxBounds) {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<false, true, 1, 3, 56><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<false, true, 3, 3, 56><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                } else {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<false, false, 1, 3, 56><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<false, false, 3, 3, 56><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                }
            }
        }  else if (threadsX == 4) {
            if (checkFilterBounds) {
                if(checkFilterIdxBounds) {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<true, true, 1, 4, 32><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<true, true, 3, 4, 32><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                } else {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<true, false, 1, 4, 32><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<true, false, 3, 4, 32><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                }
            } else {
                if(checkFilterIdxBounds) {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<false, true, 1, 4, 32><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<false, true, 3, 4, 32><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                } else {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<false, false, 1, 4, 32><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<false, false, 3, 4, 32><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                }
            }
        }  else if (threadsX == 5) {
            if (checkFilterBounds) {
                if(checkFilterIdxBounds) {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<true, true, 1, 5, 20><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<true, true, 3, 5, 20><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                } else {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<true, false, 1, 5, 20><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<true, false, 3, 5, 20><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                }
            } else {
                if(checkFilterIdxBounds) {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<false, true, 1, 5, 20><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<false, true, 3, 5, 20><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                } else {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<false, false, 1, 5, 20><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<false, false, 3, 5, 20><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                }
            }
        }  else if (threadsX == 6) {
            if (checkFilterBounds) {
                if(checkFilterIdxBounds) {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<true, true, 1, 6, 14><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<true, true, 3, 6, 14><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                } else {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<true, false, 1, 6, 14><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<true, false, 3, 6, 14><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                }
            } else {
                if(checkFilterIdxBounds) {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<false, true, 1, 6, 14><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<false, true, 3, 6, 14><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                } else {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<false, false, 1, 6, 14><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<false, false, 3, 6, 14><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                }
            }
        }  else if (threadsX == 7) {
            if (checkFilterBounds) {
                if(checkFilterIdxBounds) {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<true, true, 1, 7, 10><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<true, true, 3, 7, 10><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                } else {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<true, false, 1, 7, 10><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<true, false, 3, 7, 10><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                }
            } else {
                if(checkFilterIdxBounds) {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<false, true, 1, 7, 10><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<false, true, 3, 7, 10><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                } else {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<false, false, 1, 7, 10><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<false, false, 3, 7, 10><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                }
            }
        }  else if (threadsX == 8) {
            if (checkFilterBounds) {
                if(checkFilterIdxBounds) {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<true, true, 1, 8, 8><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<true, true, 3, 8, 8><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                } else {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<true, false, 1, 8, 8><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<true, false, 3, 8, 8><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                }
            } else {
                if(checkFilterIdxBounds) {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<false, true, 1, 8, 8><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<false, true, 3, 8, 8><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                } else {
                    if (imagesPerFilter == 1) {
                        conv2_bw_nofit_dynXYZ_2per<false, false, 1, 8, 8><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    } else {
                        conv2_bw_nofit_dynXYZ_2per<false, false, 3, 8, 8><<<grid, threads>>>(images, filters, targets, imgSize, filterSize, numFilters);
                    }
                }
            }
        }
    } else if(filterSize > 20) {
        bool checkFilterBounds = filterSize % 16 != 0;
        int blocksY = numFilters / 16, blocksX = numCases;
        dim3 grid(blocksX, blocksY);
        dim3 threads(16, 4, 8);

        if(checkFilterBounds) {
            if(imagesPerFilter == 1) {
                conv2_bw_nofit_4x16_2per<true, 1><<<grid, threads>>>(images, filters, targets, imgSize, filterSize);
            } else {
                conv2_bw_nofit_4x16_2per<true, 3><<<grid, threads>>>(images, filters, targets, imgSize, filterSize);
            }
        } else {
            if(imagesPerFilter == 1) {
                conv2_bw_nofit_4x16_2per<false, 1><<<grid, threads>>>(images, filters, targets, imgSize, filterSize);
            } else {
                conv2_bw_nofit_4x16_2per<false, 3><<<grid, threads>>>(images, filters, targets, imgSize, filterSize);
            }
        }
    } else if (filterSize > 14) {
        int blocksY = numFilters / 8, blocksX = numCases;
        dim3 grid(blocksX, blocksY);
        dim3 threads(16, 4, 8);
        if(filterSize == 15) {
            if(checkOutputBounds) {
                if(imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_1per<15, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_1per<15, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            } else {
                if(imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_1per<15, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_1per<15, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            }
        } else if(filterSize == 16) {
            if(checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_1per<16, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_1per<16, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_1per<16, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_1per<16, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            }
        } else if(filterSize == 17) {
            if(checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_1per<17, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_1per<17, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_1per<17, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_1per<17, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            }
        } else if(filterSize == 18) {
            if(checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_1per<18, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_1per<18, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_1per<18, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_1per<18, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            }
        } else if(filterSize == 19) {
            if(checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_1per<19, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_1per<19, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_1per<19, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_1per<19, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            }
        } else if(filterSize == 20) {
            if(checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_1per<20, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_1per<20, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_1per<20, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_1per<20, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            }
        }
    } else {
        int blocksY = numFilters / 16, blocksX = numCases;
        dim3 grid(blocksX, blocksY);
        dim3 threads(16, 4, 8);
//            printf("blocks x: %d, blocks y: %d\n", blocksX, blocksY);
        if (filterSize == 2) {
            if (checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<2, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<2, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<2, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<2, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            }
        } else if (filterSize == 3) {
            if (checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<3, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<3, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<3, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<3, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            }
        } else if (filterSize == 4) {
            if (checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<4, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<4, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<4, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<4, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            }
        } else if (filterSize == 5) {
            if (checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<5, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<5, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<5, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<5, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            }
        } else if (filterSize == 6) {
            if (checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<6, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<6, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<6, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<6, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            }
        } else if (filterSize == 7) {
            if (checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<7, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<7, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<7, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<7, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            }
        } else if (filterSize == 8) {
            if (checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<8, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<8, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<8, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<8, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            }
        } else if (filterSize == 9) {
            if (checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<9, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<9, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<9, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<9, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            }
        } else if (filterSize == 10) {
            if (checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<10, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<10, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<10, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<10, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            }
        } else if (filterSize == 11) {
            if (checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<11, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<11, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<11, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<11, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            }
        } else if (filterSize == 12) {
            if (checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<12, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<12, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<12, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<12, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            }
        } else if (filterSize == 13) {
            if (checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<13, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<13, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<13, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<13, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                }
            }
        } else if (filterSize == 14) {
            if (checkOutputBounds) {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<14, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<14, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            } else {
                if (imagesPerFilter == 1) {
                    conv2_bw_fit_4x16_2per<14, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv2_bw_fit_4x16_2per<14, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            }
        }
    }
    cutilCheckMsg("kernel execution failed");
}

void convolve2_bw(NVMatrix* images, NVMatrix* filters, NVMatrix* targets, int filterSize) {
    double dImgSize = sqrt(images->getNumCols());
//    double dFilterSize = sqrt(filters->getNumCols());
    assert(dImgSize == floor(dImgSize));
    // each row in "filters" corresponds to a set of filters to convolve with the images in the same row of "images"
    assert(images->getNumRows() == filters->getNumRows());
    assert(filters->getNumCols() % (filterSize*filterSize) == 0);
    //    assert(dFilterSize == floor(dFilterSize));
    int imgSize = int(dImgSize);
    //    int filterSize = int(dFilterSize);
    int numCases = images->getNumRows();
    int numFilters = filters->getNumCols() / (filterSize * filterSize);
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;

    assert(numFilters % 16 == 0);
    assert(targets->getNumElements() == numOutputs * numFilters * numCases);
    assert(!images->isTrans());
    assert(!filters->isTrans());
    assert(!targets->isTrans());
    assert(imgSize > filterSize);

    _convolve2_bw(images->getDevData(), filters->getDevData(), targets->getDevData(),
                 numCases, numFilters, imgSize, filterSize, 1);
}
/*
 * Here the "filters" might represent the activities of the hidden layer of a convolutional net
 * (the output of a convolution), so the color attribute does not apply to them.
 */
void convolve2_color(NVMatrix* images, NVMatrix* filters, NVMatrix* targets, int filterSize) {
    assert(images->getNumCols() % 3 == 0);
    assert(images->getNumRows() == filters->getNumRows());
    double dImgSize = sqrt(images->getNumCols() / 3);
    assert(dImgSize == floor(dImgSize));
    assert(filters->getNumCols() % (filterSize*filterSize) == 0);
    int imgSize = int(dImgSize);
    int numCases = images->getNumRows();
    int numFilters = filters->getNumCols() / (filterSize * filterSize);
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
//    int imgPixels = imgSize * imgSize;
//    int filterPixels = filterSize * filterSize;

    assert(numFilters % 16 == 0);
    assert(targets->getNumElements() == numOutputs * numFilters * numCases*3);
    assert(!images->isTrans());
    assert(!filters->isTrans());
    assert(!targets->isTrans());
    assert(imgSize > filterSize);

    _convolve2_bw(images->getDevData(), filters->getDevData(), targets->getDevData(),
                 numCases*3, numFilters, imgSize, filterSize, 3);
}
