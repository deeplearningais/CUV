#define BOOST_TEST_MODULE example
#include <cstdio>
#include <boost/test/included/unit_test.hpp>

#include <cuv_general.hpp>
#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>
#include <vector_ops.hpp>
#include <timing.hpp>
#include <random.hpp>

using namespace cuv;

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

#define MEASURE_TIME(MSG, OPERATION, ITERS)     \
	if(1){                                      \
		Timing tim;                             \
		for(int i=0;i<ITERS;i++){               \
			OPERATION ;                         \
		}                                       \
		tim.update(ITERS);                      \
		printf("%s [%s] took %4.4f us/pass\n", MSG, #OPERATION, 1000000.0f*tim.perf()); \
	}

struct Fix{
	static const int n = 8092; 
	dev_vector<float>  v,w;
	host_vector<float> x,z;
	Fix()
	:   v(n),w(n)
	,   x(n),z(n)
	{
		MEASURE_TIME("warmup", apply_scalar_functor(v, SF_EXP), 100);
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( vec_rnd )
{
	MEASURE_TIME("vec_rnd_normal", fill_rnd_normal(v), 100);
	MEASURE_TIME("vec_rnd_uniform", fill_rnd_uniform(v), 100);
}

BOOST_AUTO_TEST_CASE( vec_ops_exp )
{
	sequence(v);
	sequence(x);
	MEASURE_TIME("vec_ops_exp", apply_scalar_functor(v, SF_EXP), 1000);
	MEASURE_TIME("vec_ops_exp", apply_scalar_functor(x, SF_EXP), 1000);
}

BOOST_AUTO_TEST_CASE( vec_ops_unary2 )
{
	sequence(v);
	sequence(x);
	MEASURE_TIME("vec_ops_scalar_mult", apply_scalar_functor(v, SF_MULT,0.01f), 1000);
	MEASURE_TIME("vec_ops_scalar_mult", apply_scalar_functor(x, SF_MULT,0.01f), 1000);
	MEASURE_TIME("vec_ops_scalar_add",  apply_scalar_functor(v, SF_ADD,0.01f), 1000);
	MEASURE_TIME("vec_ops_scalar_add",  apply_scalar_functor(x, SF_ADD,0.01f), 1000);
}

BOOST_AUTO_TEST_CASE( vec_axpby )
{
	sequence(v);
	sequence(w);
	sequence(x);
	sequence(z);
	MEASURE_TIME("vec_ops_axpby", apply_binary_functor(v,w, BF_AXPBY,1.8f,1.2f), 1000);
	MEASURE_TIME("vec_ops_axpby", apply_binary_functor(x,z, BF_AXPBY,1.8f,1.2f), 1000);
}

BOOST_AUTO_TEST_CASE( vec_add )
{
	sequence(v);
	sequence(w);
	sequence(x);
	sequence(z);
	MEASURE_TIME("vec_ops_add", apply_binary_functor(v,w, BF_ADD), 1000);
	MEASURE_TIME("vec_ops_add", apply_binary_functor(x,z, BF_ADD), 1000);
}


BOOST_AUTO_TEST_SUITE_END()
