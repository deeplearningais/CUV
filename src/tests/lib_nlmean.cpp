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

#define BOOST_TEST_MODULE lib_intimg
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <cuv/tools/cuv_general.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/libs/nlmeans/nlmeans.hpp>
#include <cuv/tools/timing.hpp>
#include <cuv/tools/cuv_test.hpp>

#include <cuv/libs/cimg/cuv_cimg.hpp>

using namespace cuv;

#define MEASURE_TIME(MSG, OPERATION, ITERS)     \
	float MSG;                                  \
	if(1){                                      \
		Timing tim;                             \
		for(int i=0;i<ITERS;i++){               \
			OPERATION ;                         \
		}                                       \
			safeThreadSync();               \
		tim.update(ITERS);                      \
		printf("%s [%s] took %4.4f us/pass\n", #MSG, #OPERATION, 1000000.0f*tim.perf()); \
		MSG = 1000000.0f*tim.perf();            \
	}

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
	}
	~Fix(){
	}
};

BOOST_FIXTURE_TEST_SUITE( s, Fix )

/** 
 * @test
 * @brief create dense device matrix.
 */
BOOST_AUTO_TEST_CASE( test_nlmeans )
{
	tensor<float,host_memory_space,row_major> mh(extents[410][192][192]), mhorig;

#define LENA 0
#if !LENA
	FILE* pFile = fopen("/home/VI/staff/schulz/checkout/git/wurzel/data/GersteLA_192x192x410_normal-upsampled.dat","r");
	unsigned int s = fread(mh.ptr(),sizeof(float),mh.size(),pFile);
	fclose(pFile);
	mh/=425000.f;
#else
	libs::cimg::load(mh,"src/tests/data/lena_color.jpg");
	//libs::cimg::load(mh,"src/tests/data/colored_square.jpg");
	for(int i=0;i<mh.shape()[0];i++)
	for(int j=0;j<mh.shape()[1];j++)
	for(int k=0;k<mh.shape()[2];k++){
		float f = 15.f*sqrt(-2*log(drand48()))*cos(2*M_PI*drand48());
		mh(i,j,k) += f;
	}
	//tensor<float,host_memory_space,row_major> mh2(indices[0][index_range(0,300)][index_range(0,400)], mh), mh3;
	//mh3 = mh2;
	//mh  = mh3;

	libs::cimg::show(mh,"filtered image");
	//mh.reshape(extents[300][1][400]);
	mhorig = mh;
#endif

	//cuvAssert(s==mh.memsize());
	typedef tensor<float,dev_memory_space,row_major, memory2d_tag> dev_t;
	typedef tensor<float,dev_memory_space,row_major> dev1d_t;
	dev_t    md;
	md  = mh;
	dev1d_t dst(mh.shape());
#if LENA
	cuv::apply_scalar_functor(mh,cuv::SF_MAX,0.f);
	cuv::apply_scalar_functor(mh,cuv::SF_MIN,255.f);
	libs::cimg::save(mh,"noisedlena.png");
	//libs::nlmeans::filter_nlmean(dst,md, 20, 5, 15.f, 0.25f, false, true);
	//filter_nlmean(tensor dst, const tensor src, search_radius, filter_radius, sigma, dist_sigma, step_size, bool threeDim, bool verbose){
	MEASURE_TIME(dev, libs::nlmeans::filter_nlmean(dst,md, 50, 6,  15.00f, -1.f,  1.0f, false, false), 1);
	//dst.reshape(extents[300][400]);
	
#else
	libs::nlmeans::filter_nlmean(dst, md,  50, 2, 0.02f, -1.f,  1.0f,  true, true);
#endif
	mh = dst;
#if !LENA
	pFile = fopen("nlmeanresult.dat","w");
	fwrite(mh.ptr(), sizeof(float), mh.size(), pFile);
#else
	//libs::cimg::show(tensor<float,host_memory_space,row_major>(indices[0][index_range(0,400)][index_range(0,400)], mh), "channel0");
	//libs::cimg::show(tensor<float,host_memory_space,row_major>(indices[1][index_range(0,400)][index_range(0,400)], mh), "channel1");
	//libs::cimg::show(tensor<float,host_memory_space,row_major>(indices[2][index_range(0,400)][index_range(0,400)], mh), "channel2");
	libs::cimg::show(mh,"filtered image");
	libs::cimg::show(mh-mhorig,"difference");
	cuv::apply_scalar_functor(mh,cuv::SF_MAX,0.f);
	cuv::apply_scalar_functor(mh,cuv::SF_MIN,255.f);
	libs::cimg::save(mh,"filteredlena2.png");
#endif
}

BOOST_AUTO_TEST_SUITE_END()

