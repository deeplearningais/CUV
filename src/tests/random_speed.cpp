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
#include <boost/test/floating_point_comparison.hpp>


#include <timing.hpp>
#include <cuv_general.hpp>
#include <vector.hpp>
#include <vector_ops.hpp>
#include <../random/random.hpp>

#define MEASURE_TIME(MSG, OPERATION, ITERS)     \
	float MSG;                                  \
	if(1){                                      \
		Timing tim;                             \
		for(int i=0;i<ITERS;i++){               \
			printf(".");fflush(stdout);         \
			OPERATION ;                         \
		}                                       \
		tim.update(ITERS);                      \
		printf("%s [%s] took %4.4f us/pass\n", #MSG, #OPERATION, 1000000.0f*tim.perf()); \
		MSG = 1000000.0f*tim.perf();            \
	}

using namespace cuv;

struct Fix{
	vector<float,dev_memory_space> v;
	vector<float,host_memory_space> x;
	static const int n = 150*150*96;
	Fix()
		:v(n),x(n) // needs large sample number.
	{
		//initCUDA(1);
		initialize_mersenne_twister_seeds();
	}
	~Fix(){
		//exitCUDA();
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( random_uniform )
{
	MEASURE_TIME(dev, fill_rnd_uniform(v), 10);
	MEASURE_TIME(host, fill_rnd_uniform(x), 10);
	printf("Speedup: %3.4f\n", host/dev);
	BOOST_CHECK_LT(dev,host);
}
BOOST_AUTO_TEST_CASE( random_normal )
{
	apply_0ary_functor(v,NF_FILL,0);
	apply_0ary_functor(x,NF_FILL,0);	
	MEASURE_TIME(dev,add_rnd_normal(v),10);
	MEASURE_TIME(host,add_rnd_normal(x),10);
	printf("Speedup: %3.4f\n", host/dev);
	BOOST_CHECK_LT(dev,host);
}
BOOST_AUTO_TEST_CASE( binarize )
{
	fill_rnd_uniform(v);
	fill_rnd_uniform(x);
	MEASURE_TIME(dev,rnd_binarize(v),10);
	MEASURE_TIME(host,rnd_binarize(x),10);
	printf("Speedup: %3.4f\n", host/dev);
	BOOST_CHECK_LT(dev,host);
}




BOOST_AUTO_TEST_SUITE_END()
