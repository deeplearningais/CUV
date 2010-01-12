/*
 * testconv_extras.cuh
 *
 *  Created on: Nov 10, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#ifndef TESTCONV_EXTRAS_CUH_
#define TESTCONV_EXTRAS_CUH_
#include <cutil_inline.h>
#include <matrix.h>
#include <nvmatrix.cuh>
#include "conv_extras.cuh"

void test_conv_bw_fit_dyn_2per(int boardNum);
void test_conv_bw_nofit_dyn_1per(int imgSize, int filterSize, int threadsY, int threadsX, int boardNum);
void test_conv_bw_nofit_dyn_2per(int imgSize, int filterSize, int threadsY, int threadsX, int boardNum);
void test_conv_bw_nofit_4x16_dynfilter_2per(int imgSize, int filterSize, int filterCacheY, int filterCacheX, int boardNum);

#endif /* TESTCONV_EXTRAS_CUH_ */
