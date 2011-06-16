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
#include <cuv/basics/tensor.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/tools/timing.hpp>
#include <cuv/random/random.hpp>

using namespace cuv;

#define MEASURE_TIME(MSG, OPERATION, ITERS)     \
	float MSG;                                  \
	if(1){                                      \
		Timing tim;                             \
		for(int i=0;i<ITERS;i++){               \
			OPERATION ;                     \
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
	static const int n; 
	tensor<float,dev_memory_space,column_major>  u_dev,v_dev,w_dev;
	tensor<float,host_memory_space,column_major> u_host,v_host,w_host;
	Fix()
	:   u_dev(n,n),v_dev(n,n),w_dev(n,n)
	,   u_host(n,n),v_host(n,n),w_host(n,n)
	{
		//MEASURE_TIME("warmup", apply_scalar_functor(v_dev, SF_EXP), 100);
	}
	~Fix(){
	}
};
const int Fix::n = 1024;


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( mat_prod )
{
	sequence(v_dev); apply_scalar_functor(v_dev,SF_MULT,0.001f);
	sequence(w_dev); apply_scalar_functor(w_dev,SF_MULT,0.001f);
	sequence(v_host); apply_scalar_functor(v_host,SF_MULT,0.001f);
	sequence(w_host); apply_scalar_functor(w_host,SF_MULT,0.001f);
	MEASURE_TIME(dev,  prod(u_dev,v_dev,w_dev, 'n','t'), 10);
	MEASURE_TIME(host, prod(u_host,v_host,w_host, 'n','t'), 10);
	printf("Speedup: %3.4f\n", host/dev);
}

BOOST_AUTO_TEST_CASE( mat_plus_vec )
{
	sequence(v_dev);
	sequence(v_host);
	tensor<float,dev_memory_space> v_vec(n); sequence(v_vec);
	tensor<float,host_memory_space> x_vec(n); sequence(x_vec);
	MEASURE_TIME(dev,  matrix_plus_col(v_dev,v_vec), 10);
	MEASURE_TIME(host, matrix_plus_col(v_host,x_vec), 10);
	printf("Speedup: %3.4f\n", host/dev);
}

BOOST_AUTO_TEST_CASE( mat_plus_vec_row_maj )
{
	tensor<float,dev_memory_space,row_major> V_dev(v_dev.shape()); sequence(V_dev);
	tensor<float,host_memory_space,row_major> X_host(v_host.shape()); sequence(X_host);
	tensor<float,dev_memory_space>   v_vec(n); sequence(v_vec);
	tensor<float,host_memory_space>  x_vec(n); sequence(x_vec);
	MEASURE_TIME(dev,  matrix_plus_col(V_dev,v_vec), 10);
	MEASURE_TIME(host, matrix_plus_col(X_host,x_vec), 10);
	printf("Speedup: %3.4f\n", host/dev);
}

//BOOST_AUTO_TEST_CASE( mat_op_argmax )
//{	
	//tensor<float,dev_memory_space,row_major> V_dev(v_dev.shape()[0],v_dev.shape()[1]); sequence(V_dev);
	//tensor<float,host_memory_space,row_major> X_host(v_host.shape()[0],v_host.shape()[1]); sequence(X_host);
	//tensor<float,dev_memory_space>   v_vec(n); sequence(v_vec);
	//tensor<float,host_memory_space>  x_vec(n); sequence(x_vec);
	//MEASURE_TIME(dev,  argmax_to_column(v_vec,V_dev), 10);
	//MEASURE_TIME(host, argmax_to_column(x_vec,X_host), 10);

	//printf("Speedup: %3.4f\n", host/dev);

//}

//BOOST_AUTO_TEST_CASE( mat_op_argmax_new )
//{	
	//tensor<float,dev_memory_space,row_major> V_dev(v_dev.shape()[0],v_dev.shape()[1]); sequence(V_dev);
	//tensor<float,host_memory_space,row_major> X_host(v_host.shape()[0],v_host.shape()[1]); sequence(X_host);
	//tensor<float,dev_memory_space>   v_vec(n); sequence(v_vec);
	//tensor<float,host_memory_space>  x_vec(n); sequence(x_vec);
	//MEASURE_TIME(dev,  reduce_to_col(v_vec,V_dev,RF_ARGMAX), 10);
	//MEASURE_TIME(host, reduce_to_col(x_vec,X_host,RF_ARGMAX), 10);

	//printf("Speedup: %3.4f\n", host/dev);

//}

BOOST_AUTO_TEST_CASE( mat_transpose )
{
	const int n = 256;
	const int m = 4096;

	tensor<float,dev_memory_space,column_major> V_dev(n,m), W(m,n); sequence(V_dev);
	tensor<float,host_memory_space,column_major> X_host(n,m), Y(m,n); sequence(X_host);

	MEASURE_TIME(dev,  transpose(W,V_dev), 10);
	MEASURE_TIME(host, transpose(Y,X_host), 10);
	printf("Speedup: %3.4f\n", host/dev);
}

BOOST_AUTO_TEST_CASE( mat_op_reduce_big_rm_to_row )
{
	tensor<float,dev_memory_space,row_major> A_dev(32, 384*384*32);
	tensor<float,dev_memory_space> V_dev(384*384*32);
	tensor<float,host_memory_space,row_major> A_host(32, 384*384*32);
	tensor<float,host_memory_space> V_host(384*384*32);

//	sequence(A_dev);
//	sequence(V_dev);
//	sequence(A_host);
//	sequence(V_host);

	fill(A_dev, 1);
	fill(A_host, 1);

	fill(V_dev, 0);
	fill(V_host, 0);

	MEASURE_TIME(dev, reduce_to_row(V_dev,A_dev,RF_ADD, 1.0f, 1.0f), 10);
	//MEASURE_TIME(host, reduce_to_row(V_host,A_host,RF_ADD, 1.0f, 1.0f), 10);

}

BOOST_AUTO_TEST_SUITE_END()
