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
		safeThreadSync();                      \
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

BOOST_AUTO_TEST_CASE( test_reorder_for_conv )
{
    using namespace cuv::alex_conv;
	unsigned int nImgChan = 7;      // must be divisible by nGroups
	unsigned int nImgPix  = 5;
	unsigned int nImg     = 2;

    tensor<float,dev_memory_space,row_major> inp(cuv::extents[nImg][nImgChan][nImgPix][nImgPix]);
	tensor<float,dev_memory_space,row_major> src(cuv::extents[nImgChan][nImgPix][nImgPix][nImg]);

    tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nImg][nImgChan][nImgPix][nImgPix]);
	tensor<float,host_memory_space,row_major> src_h(cuv::extents[nImgChan][nImgPix][nImgPix][nImg]);

    tensor<float,dev_memory_space,row_major> inp2(cuv::extents[nImg][nImgChan][nImgPix][nImgPix]);
    tensor<float,host_memory_space,row_major> inp2_h(cuv::extents[nImg][nImgChan][nImgPix][nImgPix]);

    sequence(inp);
    sequence(inp_h);
    src = 0.f;

    MEASURE_TIME(host_reorder,cuv::alex_conv::reorder_for_conv(src_h,inp_h), 2);
    MEASURE_TIME(dev_reorder, cuv::alex_conv::reorder_for_conv(src,inp), 2);
    cuvAssert(inp.shape(1)==src.shape(0));
    cuvAssert(inp.shape(2)==src.shape(1));
    cuvAssert(inp.shape(3)==src.shape(2));
    cuvAssert(inp.shape(0)==src.shape(3));
    unsigned int cnt=0;
    for(unsigned int i=0;i<nImg;i++)
        for(unsigned int j=0;j<nImgChan;j++)
            for(unsigned int k=0;k<nImgPix;k++){
                for(unsigned int l=0;l<nImgPix;l++){
                    BOOST_CHECK_EQUAL(inp(i,j,k,l), src(j,k,l,i));
                }
            }
    MEASURE_TIME(host_reorder2,cuv::alex_conv::reorder_from_conv(inp2,src),2);
    MEASURE_TIME(dev_reorder2,cuv::alex_conv::reorder_from_conv(inp2_h,src_h),2);
    BOOST_CHECK_EQUAL(inp.ndim(),inp2.ndim());
    BOOST_CHECK_EQUAL(inp.shape(0),inp2.shape(0));
    BOOST_CHECK_EQUAL(inp.shape(1),inp2.shape(1));
    BOOST_CHECK_EQUAL(inp.shape(2),inp2.shape(2));
    BOOST_CHECK_EQUAL(inp.shape(3),inp2.shape(3));
    BOOST_CHECK_EQUAL(inp_h.ndim(),inp2_h.ndim());
    BOOST_CHECK_EQUAL(inp_h.shape(0),inp2_h.shape(0));
    BOOST_CHECK_EQUAL(inp_h.shape(1),inp2_h.shape(1));
    BOOST_CHECK_EQUAL(inp_h.shape(2),inp2_h.shape(2));
    BOOST_CHECK_EQUAL(inp_h.shape(3),inp2_h.shape(3));
    for(unsigned int i=0;i<nImg;i++)
        for(unsigned int j=0;j<nImgChan;j++)
            for(unsigned int k=0;k<nImgPix;k++){
                for(unsigned int l=0;l<nImgPix;l++){
                    BOOST_CHECK_EQUAL(inp(i,j,k,l), inp2(i,j,k,l));
                    BOOST_CHECK_EQUAL(inp_h(i,j,k,l), inp2_h(i,j,k,l));
                }
            }
}

BOOST_AUTO_TEST_CASE( test_conv2d_hostdev )
{
    using namespace cuv::alex_conv;
	unsigned int nImgChan = 8;      // must be divisible by nGroups
	unsigned int nImgPix  = 176;
	unsigned int nImg     = 32;
    unsigned int nGroups  = 1;      // must be divisible by 2 ??
   
	unsigned int nFiltChan = nImgChan/nGroups;
	unsigned int nFiltPix  = 7;
	unsigned int nFilt     = 32; 

    unsigned int nResPix   = nImgPix+1-nFiltPix;


    tensor<float,dev_memory_space,row_major> inp(cuv::extents[nImg][nImgChan][nImgPix][nImgPix]);
	tensor<float,dev_memory_space,row_major> src(cuv::extents[nImgChan][nImgPix][nImgPix][nImg]);
	tensor<float,dev_memory_space,row_major> dst(cuv::extents[nFilt][nResPix][nResPix][nImg]);
	tensor<float,dev_memory_space,row_major> flt(cuv::extents[nFiltChan][nFiltPix*nFiltPix][nFilt]);
    cuv::alex_conv::reorder_for_conv(src,inp);

    for(unsigned int i=0;i<inp.size();i++) inp[i] = -0.1 + drand48();
    for(unsigned int i=0;i<flt.size();i++) flt[i] = -0.1 + drand48();
    dst = 0.f;

	tensor<float,host_memory_space,row_major> hsrc(cuv::extents[nImgChan][nImgPix][nImgPix][nImg]);
	tensor<float,host_memory_space,row_major> hdst(cuv::extents[nFilt][nResPix][nResPix][nImg]);
	tensor<float,host_memory_space,row_major> hflt(cuv::extents[nFiltChan][nFiltPix*nFiltPix][nFilt]);
    hsrc=src;
    hdst=dst;
    hflt=flt;

    //MEASURE_TIME(conv_dev,         convolve2d(dst,src,flt, 0, 1, nGroups), 10);
    //MEASURE_TIME(conv_hst,         convolve2d(hdst,hsrc,hflt, 0, 1, nGroups), 10);

    for(unsigned int i=0;i<hdst.shape(0);i++)
        for (unsigned int j = 0; j < hdst.shape(1); ++j)
            for (unsigned int k = 0; k < hdst.shape(2); ++k)
            {
                //BOOST_CHECK_CLOSE((float)dst(i,j,k),(float)hdst(i,j,k),0.1f);
            }

    // check derivative w.r.t. images
    for(unsigned int i=0;i<hdst.size();i++) hdst[i] = -0.1 + drand48();
    hdst = 0.f; hdst[0]=1.f;
    dst = hdst; 
    flt=2.f; hflt = flt;
    src = 0.f;
    hsrc = 0.f;
    //MEASURE_TIME(d_conv_dimg_dev,           d_conv2d_dimg(src,dst,flt, 0, 1, nGroups), 10);
    //MEASURE_TIME(d_conv_dimg_hst,  hsrc=0.f;d_conv2d_dimg(hsrc,hdst,hflt, 0, 1, nGroups), 10);

    std::cout << "norm2 of gradient: "<<cuv::norm2(src)<<std::endl;

    for(unsigned int i=0;i<hsrc.shape(0);i++)
        for (unsigned int j = 0; j < hsrc.shape(1); ++j)
            for (unsigned int k = 0; k < hsrc.shape(2); ++k)
            {
                //BOOST_CHECK_CLOSE((float)src(i,j,k),(float)hsrc(i,j,k),0.01f);
            }

    MEASURE_TIME(d_conv_dfilt_dev, d_conv2d_dfilt(flt,dst,src, 0, 1, nGroups,4), 10);
    MEASURE_TIME(d_conv_dfilt_host, d_conv2d_dfilt(hflt,hdst,hsrc, 0, 1, nGroups,4), 10);
    for(unsigned int i=0;i<flt.shape(0);i++)
        for (unsigned int j = 0; j < flt.shape(1); ++j)
            for (unsigned int k = 0; k < flt.shape(2); ++k)
            {
                //BOOST_CHECK_CLOSE((float)flt(i,j,k),(float)hflt(i,j,k),0.01f);
            }
}


BOOST_AUTO_TEST_CASE( test_conv2d )
{
    using namespace cuv::alex_conv;
	unsigned int nImgChan = 8;      // must be divisible by nGroups
	unsigned int nImgPix  = 176;
	unsigned int nImg     = 32;
    unsigned int nGroups  = 1;      // must be divisible by 2 ??
   
	unsigned int nFiltChan = nImgChan/nGroups;
	unsigned int nFiltPix  = 7;
	unsigned int nFilt     = 32; 

    unsigned int nResPix   = nImgPix-nFiltPix+1;

    tensor<float,dev_memory_space,row_major> inp(cuv::extents[nImg][nImgChan][nImgPix][nImgPix]);

	tensor<float,dev_memory_space,row_major> src(cuv::extents[nImgChan][nImgPix][nImgPix][nImg]);
	tensor<float,dev_memory_space,row_major> dst(cuv::extents[nFilt][nResPix][nResPix][nImg]);

	tensor<float,dev_memory_space,row_major> flt(cuv::extents[nFiltChan][nFiltPix*nFiltPix][nFilt]);

    cuv::alex_conv::reorder_for_conv(src,inp);

    //convolve2d(tensor<float,dev_memory_space>& dst, 
    //        const tensor<float,dev_memory_space>& img, 
    //        const tensor<float,dev_memory_space>& filter,
    //        unsigned int nModulesX,
    //        unsigned int paddingStart, 
    //        unsigned int moduleStride,
    //        unsigned int nGroups){

    MEASURE_TIME(conv_dev,         convolve2d(dst,src,flt, 0, 1, nGroups), 10);
    MEASURE_TIME(d_conv_dimg_dev,  d_conv2d_dimg(src,dst,flt, 0, 1, nGroups), 10);
    MEASURE_TIME(d_conv_dfilt_dev, d_conv2d_dfilt(flt,dst,src, 0, 1, nGroups,4), 10);

}

BOOST_AUTO_TEST_CASE( test_pairwise_norm )
{
    using namespace cuv::alex_conv;
    unsigned int nImg = 2;
    unsigned int nPix = 4;
    unsigned int nChan = 8;
    tensor<float,dev_memory_space,row_major> inp(cuv::extents[nChan][nPix][nPix][nImg]);
    tensor<float,dev_memory_space,row_major> res(cuv::extents[nChan/2][nPix][nPix][nImg]);

    tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nChan][nPix][nPix][nImg]);
    tensor<float,host_memory_space,row_major> res_h(cuv::extents[nChan/2][nPix][nPix][nImg]);

    cuv::sequence(inp);
    cuv::sequence(inp_h);

    res = 0.f;
    res_h = 0.f;
    pairwise_norm(res,inp);
    pairwise_norm(res_h,inp_h);

    for(unsigned int i=0;i<nChan/2;i++)
        for(unsigned int j=0;j<nPix;j++)
            for(unsigned int k=0;k<nPix;k++){
                for(unsigned int l=0;l<nImg;l++){
                    BOOST_CHECK_EQUAL(res(i,j,k,l), res_h(i,j,k,l));

                    float s0 = inp_h(2*i+0,j,k,l);
                    float s1 = inp_h(2*i+1,j,k,l);
                    BOOST_CHECK_CLOSE(1.f, 1.f + res_h(i,j,k,l)-sqrt(s0*s0 + s1*s1), 0.01f);
                }
            }
}

BOOST_AUTO_TEST_SUITE_END()
