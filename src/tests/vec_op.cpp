#define BOOST_TEST_MODULE example
#include <boost/test/included/unit_test.hpp>

#include <cuv_general.hpp>
#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>
#include <vector_ops.hpp>

using namespace cuv;

struct Fix{
	dev_vector<float> v,w;
	Fix()
	:   v(256),w(256)
	{
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( vec_ops_unary1 )
{
	apply_scalar_functor(v, SF_EXP);
	apply_scalar_functor(v, SF_EXACT_EXP);
}

BOOST_AUTO_TEST_CASE( vec_ops_binary1 )
{
	sequence(v);
	sequence(w);
	apply_scalar_functor(v,SF_ADD,1);
	for(int i=0;i<256;i++){
		BOOST_CHECK_EQUAL(v[i], i + 1);
	}
	apply_binary_functor(v,w, BF_ADD);
	for(int i=0;i<256;i++){
		BOOST_CHECK_EQUAL(v[i], i + i + 1);
	}
}

BOOST_AUTO_TEST_CASE( vec_ops_binary2 )
{
	apply_binary_functor(v,w, BF_AXPY, 2);
}

BOOST_AUTO_TEST_CASE( vec_ops_copy )
{
	// generate data
	sequence(w);

	// copy data from v to w
	copy(v,w);
	for(int i=0;i<256;i++){
		BOOST_CHECK_EQUAL(v[i],w[i]);
	}
}

BOOST_AUTO_TEST_CASE( vec_ops_unary_add )
{
	sequence(v);
	apply_scalar_functor(v,SF_ADD,3.8f);
	for(int i=0;i<256;i++){
		BOOST_CHECK_EQUAL(v[i], i+3.8f);
	}
}
BOOST_AUTO_TEST_CASE( vec_ops_axpby )
{
	// generate data
	sequence(w);
	sequence(v);
	apply_scalar_functor(v,SF_ADD, 1.f);
	BOOST_CHECK_EQUAL(w[0],0);
	BOOST_CHECK_EQUAL(v[0],1);

	// copy data from v to w
	apply_binary_functor(v,w,BF_AXPBY, 2.f,3.f);
	for(int i=0;i<256;i++){
		BOOST_CHECK_EQUAL(v[i], 2*(i+1) + 3*i );
	}
}


BOOST_AUTO_TEST_CASE( vec_ops_0ary1 )
{
	// test sequence
	sequence(v);
	for(int i=0;i<256;i++){
		BOOST_CHECK_EQUAL(v[i], i);
	}

	// test fill
	fill(w,1);
	for(int i=0;i<256;i++){
		BOOST_CHECK_EQUAL(w[i], 1);
	}
}

BOOST_AUTO_TEST_CASE( vec_ops_norms )
{
	sequence(v);
	float f1 = norm1(v), f1_ = 0;
	float f2 = norm2(v), f2_ = 0;
	for(int i=0;i<256;i++){
		f2_ += v[i]*v[i];
		f1_ += fabs(v[i]);
	}
	f2_ = sqrt(f2_);
	BOOST_CHECK_EQUAL(f1,f1_);
	BOOST_CHECK_EQUAL(f2,f2_);
}



BOOST_AUTO_TEST_SUITE_END()
