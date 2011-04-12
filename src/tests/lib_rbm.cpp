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

#define BOOST_TEST_MODULE lib_rbm
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <iostream>
#include <cuv/tools/cuv_test.hpp>

#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/tensor.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/libs/rbm/rbm.hpp>

using namespace cuv;
using namespace cuv::libs::rbm;

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
	static const int N;
	tensor<float,host_memory_space> v;
	tensor<float,host_memory_space,column_major> m;
	Fix()
	:   v(N)
	,	m(extents[10][N]){}

	~Fix(){
	}
};

const int Fix::N=10;

BOOST_FIXTURE_TEST_SUITE( s, Fix )

BOOST_AUTO_TEST_CASE( set_bits )
{
	tensor<float,dev_memory_space,column_major> m2(m.shape());
	set_binary_sequence(m,  0);
	set_binary_sequence(m2, 0);

	MAT_CMP(m, m2, 0.001);
}
BOOST_AUTO_TEST_CASE( sigm_temp_host )
{
   fill(v,2);
   sequence(m);
   tensor<float,host_memory_space,column_major> m2(m);

   sigm_temperature(m, v);
   apply_scalar_functor(m2,SF_SIGM,2);

   MAT_CMP(m, m2, 0.001);
}
BOOST_AUTO_TEST_CASE( sigm_temp_dev )
{
   fill(v,2);
   sequence(m);
   tensor<float,dev_memory_space,column_major> m2(m);

   sigm_temperature(m, v);
   apply_scalar_functor(m2,SF_SIGM,2);

   MAT_CMP(m, m2, 0.001);
}

BOOST_AUTO_TEST_SUITE_END()
