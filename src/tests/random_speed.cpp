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
