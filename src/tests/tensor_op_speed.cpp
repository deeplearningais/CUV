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

#include <cuv/tools/cuv_general.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/tools/timing.hpp>
#include <cuv/random/random.hpp>
#include <cuv/tensor_ops/rprop.hpp>

using namespace cuv;

#define MEASURE_TIME(MSG, OPERATION, ITERS)     \
	float MSG;                                  \
	if(1){                                      \
		Timing tim;                             \
		for(int i=0;i<ITERS;i++){               \
			OPERATION ;                         \
                        safeThreadSync();               \
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
	tensor<float,dev_memory_space>  v_dev,w_dev;
	tensor<float,host_memory_space> v_host,w_host;
	Fix()
	:   v_dev(n),w_dev(n)
	,   v_host(n),w_host(n)
	{
		//MEASURE_TIME("warmup", apply_scalar_functor(v_dev, SF_EXP), 100);
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( vec_rnd )
{
	MEASURE_TIME(rnd_uniform,      fill_rnd_uniform(v_dev), 100);
	MEASURE_TIME(rnd_uniform_host, fill_rnd_uniform(v_host) , 100);
	printf("Speedup: %3.4f\n", rnd_uniform_host/rnd_uniform);

	MEASURE_TIME(rnd_normal,      add_rnd_normal(v_dev), 100);
	MEASURE_TIME(rnd_normal_host, add_rnd_normal(v_host) , 100);

	printf("Speedup: %3.4f\n", rnd_normal_host/rnd_normal);
}

BOOST_AUTO_TEST_CASE( vec_ops_exp )
{
	sequence(v_dev);
	sequence(v_host);
	MEASURE_TIME(dev , apply_scalar_functor(v_dev, SF_EXP), 1000);
	MEASURE_TIME(host, apply_scalar_functor(v_host, SF_EXP), 1000);
	printf("Speedup: %3.4f\n", host/dev);
}

BOOST_AUTO_TEST_CASE( vec_ops_unary2 )
{
	sequence(v_dev);
	sequence(v_host);
	MEASURE_TIME(mult_dev, apply_scalar_functor(v_dev, SF_MULT,0.01f), 1000);
	MEASURE_TIME(mult_host, apply_scalar_functor(v_host, SF_MULT,0.01f), 1000);
	printf("Speedup: %3.4f\n", mult_host/mult_dev);
	MEASURE_TIME(add_dev,  apply_scalar_functor(v_dev, SF_ADD,0.01f), 1000);
	MEASURE_TIME(add_host,  apply_scalar_functor(v_host, SF_ADD,0.01f), 1000);
	printf("Speedup: %3.4f\n", add_host/add_dev);
}

BOOST_AUTO_TEST_CASE( vec_xpby )
{
	sequence(v_dev);
	sequence(w_dev);
	sequence(v_host);
	sequence(w_host);
	MEASURE_TIME(dev, apply_binary_functor(v_dev,w_dev, BF_XPBY,1.8f), 1000);
	MEASURE_TIME(host, apply_binary_functor(v_host,w_host, BF_XPBY,1.8f), 1000);
	printf("Speedup: %3.4f\n", host/dev);
}

BOOST_AUTO_TEST_CASE( vec_add )
{
	sequence(v_dev);
	sequence(w_dev);
	sequence(v_host);
	sequence(w_host);
	MEASURE_TIME(dev, apply_binary_functor(v_dev,w_dev, BF_ADD), 1000);
	MEASURE_TIME(host, apply_binary_functor(v_host,w_host, BF_ADD), 1000);
	printf("Speedup: %3.4f\n", host/dev);
}


BOOST_AUTO_TEST_CASE( vec_rprop )
{
	tensor<signed char,dev_memory_space> dW_old(n);
	tensor<float,dev_memory_space>       dW_dev(n);
	tensor<float,dev_memory_space>       W_dev(n);
	tensor<float,dev_memory_space>       rate(n);
	tensor<signed char,host_memory_space> h_dW_old(n);
	tensor<float,host_memory_space>       W_host(n);
	tensor<float,host_memory_space>       dw_host(n);
	tensor<float,host_memory_space>       h_rate(n);
	sequence(dW_dev);           apply_scalar_functor(dW_dev, SF_ADD, -10.f);
	sequence(dW_old);
	fill(rate, 1.f);
	sequence(dw_host);         apply_scalar_functor(dW_dev, SF_ADD, -10.f);
	sequence(h_dW_old);
	fill(h_rate, 1.f);

	MEASURE_TIME(dev,  rprop(W_dev,dW_dev,dW_old,rate), 10);
	MEASURE_TIME(host, rprop(W_host,dw_host,h_dW_old,h_rate), 10);
	printf("Speedup: %3.4f\n", host/dev);
}


BOOST_AUTO_TEST_CASE( vec_lswd )
{
	tensor<float,dev_memory_space>       dW_dev(n);
	tensor<float,dev_memory_space>       W_dev(n);
	tensor<float,host_memory_space>       W_host(n);
	tensor<float,host_memory_space>       dw_host(n);

	MEASURE_TIME(dev,  learn_step_weight_decay(W_dev,dW_dev,1.f,0.05f), 10);
	MEASURE_TIME(host, learn_step_weight_decay(W_host,dw_host,1.f,0.05f), 10);
	printf("Speedup: %3.4f\n", host/dev);
}


BOOST_AUTO_TEST_SUITE_END()
