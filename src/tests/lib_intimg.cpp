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
#include <cuv/convert/convert.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>
#include <cuv/libs/integral_image/integral_image.hpp>
#include <cuv/tools/timing.hpp>
#include <cuv/tools/cuv_test.hpp>

using namespace cuv;
using namespace cuv::integral_img;

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
BOOST_AUTO_TEST_CASE( test_integral_image )
{
	tensor<float,host_memory_space,row_major> mh;
	typedef tensor<float,dev_memory_space,row_major> dev_t;
	//typedef tensor<float,dev_memory_space,row_major, linear_memory_tag> dev_t;
	tensor<float,host_memory_space,row_major> md(extents[3][512][128]);
	
	//libs::cimg::load(mh,"src/tests/data/lena_gray.png");
	libs::cimg::load(mh,"src/tests/data/colored_square.jpg");
	libs::cimg::show(mh,"loaded image");

	/*
	 * // larger image
	 *for(int k=0;k<md.shape()[0];k++)
	 *        for(int i=0;i<md.shape()[1];i++)
	 *                for (int j = 0; j < md.shape()[2]; ++j)
	 *                        md(k,i,j) = drand48();
	 *mh = md;
	 */

	dev_t d_m(mh);

	unsigned int w = mh.shape()[2];
	unsigned int h = mh.shape()[1];
	tensor_view<float,host_memory_space,row_major> view0h(indices[0][index_range(0,h)][index_range(0,w)],mh);
	tensor<float,host_memory_space,row_major> intimg_h(extents[w][h]); // inverted

	tensor_view<float,dev_memory_space,row_major> view0(indices[0][index_range(0,h)][index_range(0,w)],d_m);
	dev_t intimg_d(extents[w][h]); // inverted

	integral_image(intimg_d ,view0 ); // warmup

	safeThreadSync();

	MEASURE_TIME(dev,  integral_image(intimg_d ,view0 ), 80);
	MEASURE_TIME(host, integral_image(intimg_h,view0h), 80);
	printf("Speedup: %3.4f\n", host/dev);
	BOOST_CHECK_LT(dev,host);

	mh = intimg_h;
	libs::cimg::show(mh,"intimg_h");

	md = intimg_d;
	libs::cimg::show(md,"intimg_d");

	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++){
			BOOST_CHECK_CLOSE((float)mh(j,i), (float)md(j,i),0.1);
		}
	MAT_CMP(mh,md,0.1);
}

BOOST_AUTO_TEST_SUITE_END()
