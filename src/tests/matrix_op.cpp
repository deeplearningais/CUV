#define BOOST_TEST_MODULE example
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <cuv_general.hpp>
#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>
#include <convert.hpp>
#include <matrix_ops.hpp>

using namespace cuv;

struct Fix{
	dev_dense_matrix<float> u,v,w;
	host_dense_matrix<float> r,x,z;
	Fix()
	:   u(256,256),v(256,256),w(256,256)
	,   r(256,256),x(256,256),z(256,256)
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
	apply_scalar_functor(x, SF_EXP);
	apply_scalar_functor(x, SF_EXACT_EXP);
}

BOOST_AUTO_TEST_CASE( vec_ops_binary1 )
{
	sequence(v.vec());
	sequence(w.vec());
	apply_scalar_functor(v,SF_ADD,1);
	for(int i=0;i<256;i++){
		BOOST_CHECK_EQUAL(v.vec()[i], i + 1);
	}
	apply_binary_functor(v,w, BF_ADD);
	for(int i=0;i<256;i++){
		BOOST_CHECK_EQUAL(v.vec()[i], i + i + 1);
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
		BOOST_CHECK_EQUAL(v.vec()[i],w.vec()[i]);
	}
}

BOOST_AUTO_TEST_CASE( vec_ops_unary_add )
{
	sequence(v);
	apply_scalar_functor(v,SF_ADD,3.8f);
	for(int i=0;i<256;i++){
		BOOST_CHECK_EQUAL(v.vec()[i], i+3.8f);
	}
}
BOOST_AUTO_TEST_CASE( vec_ops_axpby )
{
	// generate data
	sequence(w);
	sequence(v);
	apply_scalar_functor(v,SF_ADD, 1.f);
	BOOST_CHECK_EQUAL(w.vec()[0],0);
	BOOST_CHECK_EQUAL(v.vec()[0],1);

	// copy data from v to w
	apply_binary_functor(v,w,BF_AXPBY, 2.f,3.f);
	for(int i=0;i<256;i++){
		BOOST_CHECK_EQUAL(v.vec()[i], 2*(i+1) + 3*i );
	}
}


BOOST_AUTO_TEST_CASE( vec_ops_0ary1 )
{
	// test sequence
	sequence(v.vec());
	for(int i=0;i<256;i++){
		BOOST_CHECK_EQUAL(v.vec()[i], i);
	}

	// test fill
	fill(w.vec(),1);
	for(int i=0;i<256;i++){
		BOOST_CHECK_EQUAL(w.vec()[i], 1);
	}
}

BOOST_AUTO_TEST_CASE( vec_ops_norms )
{
	sequence(v.vec());
	float f1 = norm1(v), f1_ = 0;
	float f2 = norm2(v), f2_ = 0;
	for(int i=0;i<256;i++){
		f2_ += v.vec()[i]*v.vec()[i];
		f1_ += fabs(v.vec()[i]);
	}
	f2_ = sqrt(f2_);
	BOOST_CHECK_EQUAL(f1,f1_);
	BOOST_CHECK_EQUAL(f2,f2_);
}

BOOST_AUTO_TEST_CASE( mat_op_mm )
{
	sequence(v);     apply_scalar_functor(v, SF_MULT, 0.01f);
	sequence(w);     apply_scalar_functor(w, SF_MULT, 0.01f);
	sequence(x);     apply_scalar_functor(x, SF_MULT, 0.01f);
	sequence(z);     apply_scalar_functor(z, SF_MULT, 0.01f);
	prod(u,v,w,'n','t');
	prod(r,x,z,'n','t');

	host_dense_matrix<float> u2(u.h(), u.w());
	convert(u2,u);
	for(int i=0;i<256;i++){
		for(int j=0;j<256;j++){
			BOOST_CHECK_CLOSE( u2(i,j), r(i,j), 0.01 );
		}
	}
}



BOOST_AUTO_TEST_SUITE_END()
