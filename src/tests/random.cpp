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
#include <iostream>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>


#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/vector.hpp>
#include <cuv/vector_ops/vector_ops.hpp>
#include <cuv/random/random.hpp>

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
	vector<float,dev_memory_space> v;
	vector<float,host_memory_space> x;
	static const int n =32368;
	Fix()
		:v(n),x(n) // needs large sample number.
	{
		initialize_mersenne_twister_seeds();
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( random_uniform )
{
	fill_rnd_uniform(v);
	fill_rnd_uniform(x);
	float m = mean(v);
	BOOST_CHECK_SMALL( m-0.5f, 0.01f );

	m = mean(x);
	BOOST_CHECK_SMALL( m-0.5f, 0.01f );
}
//BOOST_AUTO_TEST_CASE( random_normal_nan )
//{
	//vector<float,dev_memory_space> large(105*150*96);
	//apply_0ary_functor(large,NF_FILL,0);
	////apply_0ary_functor(x,NF_FILL,0);	
	//for(int iter=0;iter<10000;iter++){
		//add_rnd_normal(large);
		////add_rnd_normal(x);
		//BOOST_REQUIRE( ! has_nan(large) );
		//BOOST_REQUIRE( ! has_inf(large) );
		////BOOST_CHECK( ! has_nan(x) );
	//}
//}
BOOST_AUTO_TEST_CASE( random_normal )
{
	apply_0ary_functor(v,NF_FILL,0);
	apply_0ary_functor(x,NF_FILL,0);	
	add_rnd_normal(v);
	add_rnd_normal(x);
	float m   = mean(v);
	float std = std::sqrt(var(v));
	BOOST_CHECK_SMALL( m, 0.02f );
	BOOST_CHECK_SMALL( std-1.f, 0.01f );

	m   = mean(x);
	std = std::sqrt(var(x));
	BOOST_CHECK_SMALL( m, 0.02f );
	BOOST_CHECK_SMALL( std-1.f, 0.01f );
}
BOOST_AUTO_TEST_CASE( binarize )
{
	fill_rnd_uniform(v);
	fill_rnd_uniform(x);
	rnd_binarize(v);
	rnd_binarize(x);

	float m   = mean(v);
	BOOST_CHECK_SMALL( m-0.5f, 0.01f );
	for(int i = 0; i < n; ++ i) {
		BOOST_CHECK( v[i] == 0.f  || v[i] == 1.f );
	}

	m   = mean(x);
	BOOST_CHECK_SMALL( m-0.5f, 0.01f );
	for(int i = 0; i < n; ++ i) {
		BOOST_CHECK( v[i] == 0.f  || v[i] == 1.f );
	}
}




BOOST_AUTO_TEST_SUITE_END()
