#define BOOST_TEST_MODULE example
#include <cstdio>
#include <boost/test/included/unit_test.hpp>

#include <cuv_general.hpp>
#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>
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
	static const int dev = 2;
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
	dev_dense_matrix<float>  u,v,w;
	host_dense_matrix<float> r,x,z;
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

BOOST_AUTO_TEST_CASE( mat_rprop )
{
	dev_dense_matrix<signed char> dW_old(n,n);
	dev_dense_matrix<float>       dW(n,n);
	dev_dense_matrix<float>       rate(n,n);
	dev_dense_matrix<signed char> h_dW_old(n,n);
	dev_dense_matrix<float>       h_dW(n,n);
	dev_dense_matrix<float>       h_rate(n,n);
	sequence(dW);           apply_scalar_functor(dW, SF_ADD, -10.f);
	sequence(dW_old);
	fill(rate.vec(), 1.f);
	sequence(h_dW);         apply_scalar_functor(dW, SF_ADD, -10.f);
	sequence(h_dW_old);
	fill(h_rate.vec(), 1.f);

	MEASURE_TIME(dev,  cuv::rprop(dW,dW_old,rate), 10);
	MEASURE_TIME(host, cuv::rprop(h_dW,h_dW_old,h_rate), 10);
	printf("Speedup: %3.4f\n", host/dev);
}

BOOST_AUTO_TEST_SUITE_END()
