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

void _convolve_bw(float* images, float* filters, float* targets, int numCases,
                  int numFilters, int imgSize, int filterSize, int stride, bool useDynamics = false) {
    assert(stride == 1 || stride == 3);
    int numOutputsX = imgSize - filterSize + 1;
//    int numOutputs = numOutputsX*numOutputsX;
    bool checkOutputBounds = numOutputsX % 16 != 0;
    if(filterSize > 20) {
        bool checkFilterBounds = filterSize % 16 != 0;
        int blocksY = numFilters / 16, blocksX = numCases;
        dim3 grid(blocksX, blocksY);
        dim3 threads(16, 4, 8);

        if(checkFilterBounds) {
            if(stride == 1) {
                conv_bw_nofit_4x16_2per<true, 1><<<grid, threads>>>(images, filters, targets, imgSize, filterSize);
            } else {
                conv_bw_nofit_4x16_2per<true, 3><<<grid, threads>>>(images, filters, targets, imgSize, filterSize);
            }
        } else {
            if(stride == 1) {
                conv_bw_nofit_4x16_2per<false, 1><<<grid, threads>>>(images, filters, targets, imgSize, filterSize);
            } else {
                conv_bw_nofit_4x16_2per<false, 3><<<grid, threads>>>(images, filters, targets, imgSize, filterSize);
            }
        }

    } else if (filterSize > 14) {
        int blocksY = numFilters / 8, blocksX = numCases;
        dim3 grid(blocksX, blocksY);
        dim3 threads(16, 4, 8);
        if(filterSize == 15) {
            if(checkOutputBounds) {
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<15, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv_bw_fit_4x16_1per<15, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            } else {
                if(stride == 1) {
                    conv_bw_fit_4x16_1per<15, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv_bw_fit_4x16_1per<15, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            }
        } else if(filterSize == 16) {
            if(checkOutputBounds) {
                if (stride == 1) {
                    conv_bw_fit_4x16_1per<16, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv_bw_fit_4x16_1per<16, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            } else {
                if (stride == 1) {
                    conv_bw_fit_4x16_1per<16, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv_bw_fit_4x16_1per<16, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            }
        } else if(filterSize == 17) {
            if(checkOutputBounds) {
                if (stride == 1) {
                    conv_bw_fit_4x16_1per<17, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv_bw_fit_4x16_1per<17, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            } else {
                if (stride == 1) {
                    conv_bw_fit_4x16_1per<17, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv_bw_fit_4x16_1per<17, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            }
        } else if(filterSize == 18) {
            if(checkOutputBounds) {
                if (stride == 1) {
                    conv_bw_fit_4x16_1per<18, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv_bw_fit_4x16_1per<18, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            } else {
                if (stride == 1) {
                    conv_bw_fit_4x16_1per<18, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv_bw_fit_4x16_1per<18, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            }
        } else if(filterSize == 19) {
            if(checkOutputBounds) {
                if (stride == 1) {
                    conv_bw_fit_4x16_1per<19, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv_bw_fit_4x16_1per<19, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            } else {
                if (stride == 1) {
                    conv_bw_fit_4x16_1per<19, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv_bw_fit_4x16_1per<19, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            }
        } else if(filterSize == 20) {
            if(checkOutputBounds) {
                if (stride == 1) {
                    conv_bw_fit_4x16_1per<20, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv_bw_fit_4x16_1per<20, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            } else {
                if (stride == 1) {
                    conv_bw_fit_4x16_1per<20, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                } else {
                    conv_bw_fit_4x16_1per<20, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                }
            }
        }
    } else {
        if (useDynamics) {
            // later
        } else {
            int blocksY = numFilters / 16, blocksX = numCases;
            dim3 grid(blocksX, blocksY);
            dim3 threads(16, 4, 8);

            if (filterSize == 1) {
                throw "try multByScalar";
            } else if (filterSize == 2) {
                if (checkOutputBounds) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<2, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<2, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                } else {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<2, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<2, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                }
            } else if (filterSize == 3) {
                if (checkOutputBounds) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<3, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<3, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                } else {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<3, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<3, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                }
            } else if (filterSize == 4) {
                if (checkOutputBounds) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<4, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<4, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                } else {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<4, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<4, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                }
            } else if (filterSize == 5) {
                if (checkOutputBounds) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<5, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<5, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                } else {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<5, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<5, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                }
            } else if (filterSize == 6) {
                if (checkOutputBounds) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<6, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<6, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                } else {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<6, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<6, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                }
            } else if (filterSize == 7) {
                if (checkOutputBounds) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<7, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<7, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                } else {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<7, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<7, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                }
            } else if (filterSize == 8) {
                if (checkOutputBounds) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<8, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<8, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                } else {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<8, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<8, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                }
            } else if (filterSize == 9) {
                if (checkOutputBounds) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<9, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<9, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                } else {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<9, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<9, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                }
            } else if (filterSize == 10) {
                if (checkOutputBounds) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<10, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<10, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                } else {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<10, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<10, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                }
            } else if (filterSize == 11) {
                if (checkOutputBounds) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<11, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<11, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                } else {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<11, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<11, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                }
            } else if (filterSize == 12) {
                if (checkOutputBounds) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<12, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<12, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                } else {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<12, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<12, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                }
            } else if (filterSize == 13) {
                if (checkOutputBounds) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<13, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<13, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                } else {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<13, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<13, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);

                    }
                }
            } else if (filterSize == 14) {
                if (checkOutputBounds) {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<14, true, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<14, true, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                    }
                } else {
                    if (stride == 1) {
                        conv_bw_fit_4x16_2per<14, false, 1><<<grid, threads>>>(images, filters, targets, imgSize);
                    } else {
                        conv_bw_fit_4x16_2per<14, false, 3><<<grid, threads>>>(images, filters, targets, imgSize);
                    }
                }
            }
        }
    }
    cutilCheckMsg("kernel execution failed");
}

void convolve_bw(NVMatrix* images, NVMatrix* filters, NVMatrix* targets) {
    double dImgSize = sqrt(images->getNumCols());
    double dFilterSize = sqrt(filters->getNumCols());
    assert(dImgSize == floor(dImgSize));
    assert(dFilterSize == floor(dFilterSize));
    int imgSize = int(dImgSize);
    int filterSize = int(dFilterSize);
    int numCases = images->getNumRows();
    int numFilters = filters->getNumRows();
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;

    assert(numFilters % 16 == 0);
    assert(targets->getNumElements() == numOutputs * numFilters * numCases);
    assert(!images->isTrans());
    assert(!filters->isTrans());
    assert(!targets->isTrans());
    assert(imgSize >= filterSize);

    _convolve_bw(images->getDevData(), filters->getDevData(), targets->getDevData(),
                 numCases, numFilters, imgSize, filterSize, 1);
}

void convolve_color(NVMatrix* images, NVMatrix* filters, NVMatrix* targets) {
    assert(images->getNumCols() % 3 == 0);
    assert(filters->getNumCols() % 3 == 0);
    double dImgSize = sqrt(images->getNumCols() / 3);
    double dFilterSize = sqrt(filters->getNumCols() / 3);
    assert(dImgSize == floor(dImgSize));
    assert(dFilterSize == floor(dFilterSize));
    int imgSize = int(dImgSize);
    int filterSize = int(dFilterSize);
    int numCases = images->getNumRows();
    int numFilters = filters->getNumRows();
    int numOutputsX = imgSize - filterSize + 1;
    int numOutputs = numOutputsX * numOutputsX;
    int imgPixels = imgSize * imgSize;
    int filterPixels = filterSize * filterSize;

    assert(numFilters % 16 == 0);
    assert(targets->getNumElements() == numOutputs * numFilters * numCases);
    assert(!images->isTrans());
    assert(!filters->isTrans());
    assert(!targets->isTrans());
    assert(imgSize > filterSize);

    targets->apply(NVMatrix::ZERO);
    _convolve_bw(images->getDevData(), filters->getDevData(), targets->getDevData(),
                 numCases, numFilters, imgSize, filterSize, 3);
    _convolve_bw(images->getDevData() + imgPixels, filters->getDevData() + filterPixels, targets->getDevData(),
                 numCases, numFilters, imgSize, filterSize, 3);
    _convolve_bw(images->getDevData() + 2*imgPixels, filters->getDevData() + 2*filterPixels, targets->getDevData(),
                 numCases, numFilters, imgSize, filterSize, 3);
}
