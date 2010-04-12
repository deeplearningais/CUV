#define BOOST_TEST_MODULE example
#include <iostream>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>


#include <cuv_general.hpp>
#include <vector.hpp>
#include <vector_ops.hpp>
#include <../random/random.hpp>

using namespace cuv;

struct Fix{
	vector<float,dev_memory_space> v;
	vector<float,host_memory_space> x;
	static const int n = 8092;
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
	fill_rnd_uniform(v);
	fill_rnd_uniform(x);
	float m = mean(v);
	BOOST_CHECK_SMALL( m-0.5f, 0.01f );

	m = mean(x);
	BOOST_CHECK_SMALL( m-0.5f, 0.01f );
}
//BOOST_AUTO_TEST_CASE( random_normal_nan )
//{
	//vector<float,dev_memory_space> large(105*150*96);
	//apply_0ary_functor(large,NF_FILL,0);
	////apply_0ary_functor(x,NF_FILL,0);	
	//for(int iter=0;iter<10000;iter++){
		//add_rnd_normal(large);
		////add_rnd_normal(x);
		//BOOST_REQUIRE( ! has_nan(large) );
		//BOOST_REQUIRE( ! has_inf(large) );
		////BOOST_CHECK( ! has_nan(x) );
	//}
//}
BOOST_AUTO_TEST_CASE( random_normal )
{
	apply_0ary_functor(v,NF_FILL,0);
	apply_0ary_functor(x,NF_FILL,0);	
	add_rnd_normal(v);
	add_rnd_normal(x);
	float m   = mean(v);
	float std = std::sqrt(var(v));
	BOOST_CHECK_SMALL( m, 0.02f );
	BOOST_CHECK_SMALL( std-1.f, 0.01f );

	m   = mean(x);
	std = std::sqrt(var(x));
	BOOST_CHECK_SMALL( m, 0.02f );
	BOOST_CHECK_SMALL( std-1.f, 0.01f );
}
BOOST_AUTO_TEST_CASE( binarize )
{
	fill_rnd_uniform(v);
	fill_rnd_uniform(x);
	rnd_binarize(v);
	rnd_binarize(x);

	float m   = mean(v);
	BOOST_CHECK_SMALL( m-0.5f, 0.01f );
	for(int i = 0; i < n; ++ i) {
		BOOST_CHECK( v[i] == 0  || v[i] == 1 );
	}

	m   = mean(x);
	BOOST_CHECK_SMALL( m-0.5f, 0.01f );
	for(int i = 0; i < n; ++ i) {
		BOOST_CHECK( v[i] == 0  || v[i] == 1 );
	}
}




BOOST_AUTO_TEST_SUITE_END()
