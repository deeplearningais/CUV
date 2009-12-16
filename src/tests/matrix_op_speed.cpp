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
	dev_dense_matrix<float>  v,w;
	host_dense_matrix<float> x,z;
	Fix()
	:   v(n,n),w(n,n)
	,   x(n,n),z(n,n)
	{
		//MEASURE_TIME("warmup", apply_scalar_functor(v, SF_EXP), 100);
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( vec_xpby )
{
	sequence(v);
	sequence(w);
	sequence(x);
	sequence(z);
	MEASURE_TIME(dev,  prod(v,v,w, 'n','t'), 10);
	MEASURE_TIME(host, prod(x,x,z, 'n','t'), 10);
	printf("Speedup: %3.4f\n", host/dev);
}

BOOST_AUTO_TEST_SUITE_END()
