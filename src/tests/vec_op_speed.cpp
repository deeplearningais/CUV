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

#include <cuv_general.hpp>
#include <dense_matrix.hpp>
#include <vector_ops.hpp>
#include <timing.hpp>
#include <random.hpp>
#include <vector_ops/rprop.hpp>

using namespace cuv;

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

struct MyConfig {
	static const int dev = CUDA_TEST_DEVICE;
	MyConfig()   { 
		printf("Testing on device=%d\n",dev);
		initCUDA(dev); 
		initialize_mersenne_twister_seeds();
	}
	~MyConfig()  { exitCUDA();  }
};


BOOST_GLOBAL_FIXTURE( MyConfig );


struct Fix{
	static const int n = 784*2048; 
	vector<float,dev_memory_space>  v,w;
	vector<float,host_memory_space> x,z;
	Fix()
	:   v(n),w(n)
	,   x(n),z(n)
	{
		//MEASURE_TIME("warmup", apply_scalar_functor(v, SF_EXP), 100);
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( vec_rnd )
{
	MEASURE_TIME(rnd_uniform,      fill_rnd_uniform(v), 100);
	MEASURE_TIME(rnd_uniform_host, fill_rnd_uniform(x) , 100);
	printf("Speedup: %3.4f\n", rnd_uniform_host/rnd_uniform);

	MEASURE_TIME(rnd_normal,      add_rnd_normal(v), 100);
	MEASURE_TIME(rnd_normal_host, add_rnd_normal(x) , 100);

	printf("Speedup: %3.4f\n", rnd_normal_host/rnd_normal);
}

BOOST_AUTO_TEST_CASE( vec_ops_exp )
{
	sequence(v);
	sequence(x);
	MEASURE_TIME(dev , apply_scalar_functor(v, SF_EXP), 1000);
	MEASURE_TIME(host, apply_scalar_functor(x, SF_EXP), 1000);
	printf("Speedup: %3.4f\n", host/dev);
}

BOOST_AUTO_TEST_CASE( vec_ops_unary2 )
{
	sequence(v);
	sequence(x);
	MEASURE_TIME(mult_dev, apply_scalar_functor(v, SF_MULT,0.01f), 1000);
	MEASURE_TIME(mult_host, apply_scalar_functor(x, SF_MULT,0.01f), 1000);
	printf("Speedup: %3.4f\n", mult_host/mult_dev);
	MEASURE_TIME(add_dev,  apply_scalar_functor(v, SF_ADD,0.01f), 1000);
	MEASURE_TIME(add_host,  apply_scalar_functor(x, SF_ADD,0.01f), 1000);
	printf("Speedup: %3.4f\n", add_host/add_dev);
}

BOOST_AUTO_TEST_CASE( vec_xpby )
{
	sequence(v);
	sequence(w);
	sequence(x);
	sequence(z);
	MEASURE_TIME(dev, apply_binary_functor(v,w, BF_XPBY,1.8f), 1000);
	MEASURE_TIME(host, apply_binary_functor(x,z, BF_XPBY,1.8f), 1000);
	printf("Speedup: %3.4f\n", host/dev);
}

BOOST_AUTO_TEST_CASE( vec_add )
{
	sequence(v);
	sequence(w);
	sequence(x);
	sequence(z);
	MEASURE_TIME(dev, apply_binary_functor(v,w, BF_ADD), 1000);
	MEASURE_TIME(host, apply_binary_functor(x,z, BF_ADD), 1000);
	printf("Speedup: %3.4f\n", host/dev);
}


BOOST_AUTO_TEST_CASE( vec_rprop )
{
	vector<signed char,dev_memory_space> dW_old(n);
	vector<float,dev_memory_space>       dW(n);
	vector<float,dev_memory_space>       W(n);
	vector<float,dev_memory_space>       rate(n);
	vector<signed char,host_memory_space> h_dW_old(n);
	vector<float,host_memory_space>       h_W(n);
	vector<float,host_memory_space>       h_dW(n);
	vector<float,host_memory_space>       h_rate(n);
	sequence(dW);           apply_scalar_functor(dW, SF_ADD, -10.f);
	sequence(dW_old);
	fill(rate, 1.f);
	sequence(h_dW);         apply_scalar_functor(dW, SF_ADD, -10.f);
	sequence(h_dW_old);
	fill(h_rate, 1.f);

	MEASURE_TIME(dev,  cuv::rprop(W,dW,dW_old,rate), 10);
	MEASURE_TIME(host, cuv::rprop(h_W,h_dW,h_dW_old,h_rate), 10);
	printf("Speedup: %3.4f\n", host/dev);
}


BOOST_AUTO_TEST_CASE( vec_lswd )
{
	vector<float,dev_memory_space>       dW(n);
	vector<float,dev_memory_space>       W(n);
	vector<float,host_memory_space>       h_W(n);
	vector<float,host_memory_space>       h_dW(n);

	MEASURE_TIME(dev,  cuv::learn_step_weight_decay(W,dW,1.f,0.05f), 10);
	MEASURE_TIME(host, cuv::learn_step_weight_decay(h_W,h_dW,1.f,0.05f), 10);
	printf("Speedup: %3.4f\n", host/dev);
}


BOOST_AUTO_TEST_SUITE_END()
