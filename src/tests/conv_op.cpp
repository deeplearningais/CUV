//*LB*
// Copyright (c) 2010, University of Bonn, Institute for Computer Science VI
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
//  * Neither the name of the University of Bonn 
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





#define BOOST_TEST_MODULE example
#include <cstdio>
#include <boost/test/included/unit_test.hpp>
#include <float.h>

#include <cuv/tools/cuv_test.hpp>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/tools/timing.hpp>
#include <cuv/basics/tensor.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>
#include <cuv/random/random.hpp>
#include <cuv/convert/convert.hpp>

#define MEASURE_TIME(MSG, OPERATION, ITERS)     \
	float MSG;                                  \
	if(1){                                      \
		Timing tim;                             \
		for(int i=0;i<ITERS;i++){               \
			OPERATION ;                         \
		}                                       \
		tim.update(ITERS);                      \
		printf("%s [%s] took %4.4f us/pass\n", #MSG, #OPERATION, 1000000.0f*tim.perf()); \
		MSG = 1000000.0f*tim.perf();            \
	}

using namespace cuv;

struct MyConfig {
	static const int dev = CUDA_TEST_DEVICE;
	MyConfig()   { 
		printf("Testing on device=%d\n",dev);
		initCUDA(dev); 
	}
	~MyConfig()  { exitCUDA();  }
};


BOOST_GLOBAL_FIXTURE( MyConfig );

struct Fix{
	Fix()
	{
		//MEASURE_TIME("warmup", apply_scalar_functor(v, SF_EXP), 100);
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )

BOOST_AUTO_TEST_CASE( test_conv2d )
{
	unsigned int nImgChan = 8;      // must be divisible by nGroups
	unsigned int nImgPix  = 176;
	unsigned int nImg     = 16;
    unsigned int nGroups  = 1;      // must be divisible by 2 ??
   
    // we must set nGroups>1, so each filter will only be applied to nImgChan/nGroups inputs
	unsigned int nFiltChan = nImgChan/nGroups;
	unsigned int nFiltPix  = 7;
	unsigned int nFilt     = 32; 

    unsigned int nResPix   = nImgPix-nFiltPix+1;

	tensor<float,dev_memory_space,row_major> src(cuv::extents[nImgChan][nImgPix*nImgPix][nImg]);
	tensor<float,dev_memory_space,row_major> dst(cuv::extents[nFilt][nResPix*nResPix][nImg]);

	tensor<float,dev_memory_space,row_major> flt(cuv::extents[nFiltChan][nFiltPix*nFiltPix][nFilt]);

    //convolve2d(tensor<float,dev_memory_space>& dst, 
    //        const tensor<float,dev_memory_space>& img, 
    //        const tensor<float,dev_memory_space>& filter,
    //        unsigned int nModulesX,
    //        unsigned int paddingStart, 
    //        unsigned int moduleStride,
    //        unsigned int nGroups){

    {
        tensor<float,dev_memory_space,row_major> src0(cuv::extents[nImg][nImgChan*nImgPix*nImgPix]);  // this is what we can get from preprocessing
        tensor<float,dev_memory_space,row_major> src1(cuv::extents[nImgChan*nImgPix*nImgPix][nImg]);  // this is what we want in the end
        MEASURE_TIME(trn_dev,  transpose(src1,src0), 10);
    }

    MEASURE_TIME(conv_dev,         convolve2d(dst,src,flt, 0, 1, nGroups), 10);
    MEASURE_TIME(d_conv_dimg_dev,  d_conv2d_dimg(src,dst,flt, 0, 1, nGroups), 10);
    MEASURE_TIME(d_conv_dfilt_dev, d_conv2d_dfilt(flt,dst,src, 0, 1, nGroups,4), 10);

}

BOOST_AUTO_TEST_SUITE_END()
