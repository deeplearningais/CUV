#define BOOST_TEST_MODULE example
#include <cstdio>
#include <boost/test/included/unit_test.hpp>

#include <cuv_general.hpp>
#include <dense_matrix.hpp>
#include <vector_ops.hpp>
#include <matrix_ops.hpp>
#include <timing.hpp>
#include <random.hpp>
#include <matrix_ops/rprop.hpp>

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
	static const int n = 1024; 
	dense_matrix<float,column_major,dev_memory_space>  u,v,w;
	dense_matrix<float,column_major,host_memory_space> r,x,z;
	Fix()
	:   u(n,n),v(n,n),w(n,n)
	,   r(n,n),x(n,n),z(n,n)
	{
		//MEASURE_TIME("warmup", apply_scalar_functor(v, SF_EXP), 100);
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( mat_prod )
{
	sequence(v); apply_scalar_functor(v,SF_MULT,0.001f);
	sequence(w); apply_scalar_functor(w,SF_MULT,0.001f);
	sequence(x); apply_scalar_functor(x,SF_MULT,0.001f);
	sequence(z); apply_scalar_functor(z,SF_MULT,0.001f);
	MEASURE_TIME(dev,  prod(u,v,w, 'n','t'), 10);
	MEASURE_TIME(host, prod(r,x,z, 'n','t'), 10);
	printf("Speedup: %3.4f\n", host/dev);
}

BOOST_AUTO_TEST_CASE( mat_plus_vec )
{
	sequence(v);
	sequence(x);
	vector<float,dev_memory_space> v_vec(n); sequence(v_vec);
	vector<float,host_memory_space> x_vec(n); sequence(x_vec);
	MEASURE_TIME(dev,  matrix_plus_col(v,v_vec), 10);
	MEASURE_TIME(host, matrix_plus_col(x,x_vec), 10);
	printf("Speedup: %3.4f\n", host/dev);
}

BOOST_AUTO_TEST_CASE( mat_plus_vec_row_maj )
{
	dense_matrix<float,row_major,dev_memory_space> V(v.h(),v.w()); sequence(V);
	dense_matrix<float,row_major,host_memory_space> X(x.h(),x.w()); sequence(X);
	vector<float,dev_memory_space>   v_vec(n); sequence(v_vec);
	vector<float,host_memory_space>  x_vec(n); sequence(x_vec);
	MEASURE_TIME(dev,  matrix_plus_col(V,v_vec), 10);
	MEASURE_TIME(host, matrix_plus_col(X,x_vec), 10);
	printf("Speedup: %3.4f\n", host/dev);
}

BOOST_AUTO_TEST_CASE( mat_transpose )
{
	const int n = 256;
	const int m = 4096;

	dense_matrix<float,column_major,dev_memory_space> V(n,m), W(m,n); sequence(V);
	dense_matrix<float,column_major,host_memory_space> X(n,m), Y(m,n); sequence(X);

	MEASURE_TIME(dev,  transpose(W,V), 10);
	MEASURE_TIME(host, transpose(Y,X), 10);
	printf("Speedup: %3.4f\n", host/dev);
}

BOOST_AUTO_TEST_SUITE_END()
