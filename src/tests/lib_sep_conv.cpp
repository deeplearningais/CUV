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
#include <boost/test/included/unit_test.hpp>

#include <cuv/tools/cuv_general.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/libs/separable_conv/separable_convolution.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>

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
	}
	~Fix(){
	}
};

BOOST_FIXTURE_TEST_SUITE( s, Fix )

/** 
 * @test
 * @brief create dense device matrix.
 */
BOOST_AUTO_TEST_CASE( separable_convolution )
{
	tensor<float,host_memory_space,row_major> m;
	typedef tensor<float,dev_memory_space,row_major, memory2d_tag> dev_t;
	//typedef tensor<float,dev_memory_space,row_major, linear_memory_tag> dev_t;
	
	//libs::cimg::load(m,"src/tests/data/lena_gray.png");
	libs::cimg::load(m,"src/tests/data/colored_square.jpg");
	libs::cimg::show(m,"before smoothing");

	//dev_t d_m(extents[100][100]);
	//for(int i=0;i<100;i++)
		//for (int j = 0; j < 100; ++j)
			//d_m(i,j) = 0.f;
	//for(int i=0;i<100;i++){
		//d_m(50,i) = 1.f;
		//d_m(i,50) = 1.f;
	//}
	dev_t d_m;
       	d_m = m;

	m=d_m;
	libs::cimg::show(m,"before");

	tensor<float,dev_memory_space,row_major,memory2d_tag> view0(indices[0][index_range(0,400)][index_range(0,400)],d_m);
	tensor<float,dev_memory_space,row_major,memory2d_tag> view1(indices[1][index_range(0,400)][index_range(0,400)],d_m);
	tensor<float,dev_memory_space,row_major,memory2d_tag> view2(indices[2][index_range(0,400)][index_range(0,400)],d_m);

	dev_t gauss, sobel0, sobel1;
	sep_conv::convolve(gauss, d_m,8,sep_conv::SP_GAUSS, 2, 3.f);
	sep_conv::convolve(sobel0,gauss,8,sep_conv::SP_CENTERED_DERIVATIVE,0);
	sep_conv::convolve(sobel1,gauss,8,sep_conv::SP_CENTERED_DERIVATIVE,1);

	m=gauss;
	libs::cimg::show(m,"gauss");

	m=sobel0;
	libs::cimg::show(m,"sobel 0");

	m=sobel1;
	libs::cimg::show(m,"sobel 1");
}

BOOST_AUTO_TEST_SUITE_END()
