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
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/libs/hog/hog.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>
#include <cuv/libs/separable_conv/separable_convolution.hpp>

using namespace cuv;

struct MyConfig {
	static const int dev = CUDA_TEST_DEVICE;
	MyConfig()   { 
		printf("Testing on device=%d\n",dev);
		//initCUDA(dev); 
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
BOOST_AUTO_TEST_CASE( cuv_hog )
{
	tensor<float,host_memory_space,row_major> m;
	typedef tensor<float,dev_memory_space> dev_t;
	static const int bincnt = 9;
	
	//libs::cimg::load(m,"src/tests/data/colored_square.jpg");
	libs::cimg::load(m,"src/tests/data/lena_gray.png");
	unsigned int h = m.shape()[1];
	unsigned int w = m.shape()[2];
	dev_t md=m;

	dev_t bins(cuv::extents[bincnt][h][w]);

	libs::hog::hog(bins,md,6);

	m = bins;
	m.reshape(extents[bincnt*h][w]);
	m -= cuv::minimum(m);
	m *= 255.f / cuv::maximum(m);
	std::cout << DBG(cuv::maximum(m));
	std::cout << DBG(cuv::minimum(m));
	//libs::cimg::show(m, "bins.png");
	libs::cimg::save(m, "/tmp/bins.png");
}

BOOST_AUTO_TEST_SUITE_END()
