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
#include <limits>

#include <cuv/tools/cuv_general.hpp>
#include <cuv/random/random.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/tensor_ops/rprop.hpp>

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

template<class M>
void lswd_optimization(int N){
	tensor<float,M>       W(N);
	tensor<float,M>       dW(N);

	// we will optimize the function f(W) = 2 * (W-optimum)^2
	tensor<float,M>       optimum(N);
	fill_rnd_uniform(optimum);

	// start at random value
	fill_rnd_uniform(W);

	
	for (int iter = 0; iter < 1000; ++iter) {
		dW = optimum-W;
		learn_step_weight_decay(W,dW,0.01,0.0);
	}

	tensor<float,M>       f = (W-optimum);
	f *= f;
	f *= 2.f;
	double error = mean(f);
	BOOST_CHECK_CLOSE((float)error+1.f,(float)1.f,0.01f);


	for(int i=0;i<N;i++){
	       BOOST_CHECK_CLOSE((float)W[i]+1.f,(float)optimum[i]+1.f,0.01f);
	}
}

template<class M>
void rprop_optimization(int N){
	tensor<signed char,M> dW_old(N);
	tensor<float,M>       W(N);
	tensor<float,M>       dW(N);
	tensor<float,M>       rate(N);

	// we will optimize the function f(W) = 2 * (W-optimum)^2
	tensor<float,M>       optimum(N);
	fill_rnd_uniform(optimum);

	// start at random value
	fill_rnd_uniform(W);
	fill(dW_old, 0);     // initialize gradient
	fill(rate, 0.001f); // initialize learning rates
	
	for (int iter = 0; iter < 300; ++iter) {
		dW = optimum-W;
		rprop(W,dW,dW_old,rate);
	}

	tensor<float,M>       f = (W-optimum);
	f *= f;
	f *= 2.f;
	double error = mean(f);
	BOOST_CHECK_CLOSE(error+1.0,1.0,0.001);

	for(int i=0;i<N;i++){
	       BOOST_CHECK_CLOSE((float)W[i] + 1.f,(float)optimum[i]+ 1.f,0.01f);
	}
}


struct Fix{
	static const int N = 8092;
	Fix()
	{
		initialize_mersenne_twister_seeds();
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )

BOOST_AUTO_TEST_CASE( test_rprop_optimization_host )
{
	rprop_optimization<host_memory_space>(N);
}

BOOST_AUTO_TEST_CASE( test_rprop_optimization_dev )
{
	rprop_optimization<dev_memory_space>(N);
}

BOOST_AUTO_TEST_CASE( test_lswd_optimization_host )
{
	lswd_optimization<host_memory_space>(N);
}

BOOST_AUTO_TEST_CASE( test_lswd_optimization_dev )
{
	lswd_optimization<dev_memory_space>(N);
}


BOOST_AUTO_TEST_SUITE_END()
