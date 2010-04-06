#define BOOST_TEST_MODULE example
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <limits>

#include <cuv_general.hpp>
#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>
#include <vector_ops.hpp>
#include <vector_ops/rprop.hpp>

using namespace cuv;

struct Fix{
	dev_vector<float> v,w;
	static const int N = 8092;
	Fix()
	:   v(N),w(N)
	{
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( vec_ops_unary1 )
{
	apply_scalar_functor(v, SF_EXP);
	//apply_scalar_functor(v, SF_EXACT_EXP);
}

BOOST_AUTO_TEST_CASE( vec_ops_binary1 )
{
	sequence(v);
	sequence(w);
	apply_scalar_functor(v,SF_ADD,1);
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v[i], i + 1);
	}
	apply_binary_functor(v,w, BF_ADD);
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v[i], i + i + 1);
	}
}

BOOST_AUTO_TEST_CASE( vec_ops_binary2 )
{
	apply_binary_functor(v,w, BF_AXPY, 2);
}

BOOST_AUTO_TEST_CASE( vec_ops_scalar_2param )
{
	sequence(v);
	apply_scalar_functor(v, SF_TANH, 3, 5);
	for(int i=0;i<N;i++){
		BOOST_CHECK_CLOSE(v[i], (float)(3*tanh(5*i)), 0.1);
	}
}

BOOST_AUTO_TEST_CASE( vec_ops_copy )
{
	// generate data
	sequence(w);

	// copy data from v to w
	copy(v,w);
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v[i],w[i]);
	}
}

BOOST_AUTO_TEST_CASE( vec_ops_unary_add )
{
	sequence(v);
	apply_scalar_functor(v,SF_ADD,3.8f);
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v[i], i+3.8f);
	}
}
BOOST_AUTO_TEST_CASE( vec_ops_axpby )
{
	// generate data
	sequence(w);
	sequence(v);
	apply_scalar_functor(v,SF_ADD, 1.f);
	BOOST_CHECK_CLOSE(w[0],0.f,0.1f);
	BOOST_CHECK_CLOSE(v[0],1.f,0.1f);

	// copy data from v to w
	apply_binary_functor(v,w,BF_AXPBY, 2.f,3.f);
	for(int i=0;i<N;i++){
		BOOST_CHECK_CLOSE(v[i], 2.f*(i+1.f) + 3.f*i, 0.1f );
	}
}


BOOST_AUTO_TEST_CASE( vec_ops_0ary1 )
{
	// test sequence
	sequence(v);
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v[i], i);
	}

	// test fill
	fill(w,1);
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(w[i], 1);
	}
}

BOOST_AUTO_TEST_CASE( vec_ops_lswd )
{
	sequence(v);
	dev_vector<float> v_old(N);
	copy(v_old,v);
	sequence(w);
	host_vector<float> v2(N); sequence(v2);
	host_vector<float> w2(N); sequence(w2);
	learn_step_weight_decay(v,w,0.1f,0.05f);
	learn_step_weight_decay(v2,w2,0.1f,0.05f);
	for(int i=0;i<N;i++){
		BOOST_CHECK_CLOSE(v[i],v2[i],0.01);
		BOOST_CHECK_CLOSE(v[i],v_old[i] + 0.1f *(w[i] - 0.05f *v_old[i]),0.01);
	}

}
BOOST_AUTO_TEST_CASE( vec_ops_has_inf )
{
	host_vector<float> v2(N); fill(v2,0);
	fill(v,0);
	bool no  = has_inf(v);
	bool no2 = has_inf(v2);
	BOOST_CHECK_EQUAL(no,false);
	BOOST_CHECK_EQUAL(no2,false);
	if(std::numeric_limits<float>::has_infinity){
		v.set(3,std::numeric_limits<float>::infinity());
		v2.set(3,std::numeric_limits<float>::infinity());
		bool yes = has_inf(v);
		bool yes2 = has_inf(v2);
		BOOST_CHECK_EQUAL(yes,true);
		BOOST_CHECK_EQUAL(yes2,true);
	}else{
		BOOST_MESSAGE("Warning: we do not have Inf, skip test!");
	}
}
BOOST_AUTO_TEST_CASE( vec_ops_is_nan )
{
	host_vector<float> v2(N); fill(v2,0);
	fill(v,0);
	bool no  = has_nan(v);
	bool no2 = has_nan(v2);
	BOOST_CHECK_EQUAL(no,false);
	BOOST_CHECK_EQUAL(no2,false);
	if(std::numeric_limits<float>::has_quiet_NaN){
		v.set(3,std::numeric_limits<float>::quiet_NaN());
		v2.set(3,std::numeric_limits<float>::quiet_NaN());
		bool yes = has_nan(v);
		bool yes2 = has_nan(v2);
		BOOST_CHECK_EQUAL(yes,true);
		BOOST_CHECK_EQUAL(yes2,true);
	}else{
		BOOST_MESSAGE("Warning: we do not have NaN, skip test!");
	}
}
BOOST_AUTO_TEST_CASE( vec_ops_norms )
{
	sequence(v);
	float f1 = norm1(v), f1_ = 0;
	float f2 = norm2(v), f2_ = 0;
	for(int i=0;i<N;i++){
		f2_ += v[i]*v[i];
		f1_ += fabs(v[i]);
	}
	f2_ = sqrt(f2_);
	BOOST_CHECK_CLOSE(f1,f1_,0.1f);
	BOOST_CHECK_CLOSE(f2,f2_,0.1f);
}

BOOST_AUTO_TEST_CASE( vec_rprop )
{
	dev_vector<signed char> dW_old(N);
	dev_vector<float>       W(N);
	dev_vector<float>       dW(N);
	dev_vector<float>       rate(N);
	fill(W,0.f);
	sequence(dW);           apply_scalar_functor(dW, SF_ADD, -256.f);
	sequence(dW_old);       
	fill(rate, 1.f);
	rprop(W,dW,dW_old,rate);

	host_vector<signed char> h_dW_old(N);
	host_vector<float>       h_W(N);
	host_vector<float>       h_dW(N);
	host_vector<float>       h_rate(N);
	fill(h_W,0.f);
	sequence(h_dW);         apply_scalar_functor(h_dW, SF_ADD, -256.f);
	sequence(h_dW_old);     
	fill(h_rate, 1.f);
	rprop(h_W,h_dW,h_dW_old,h_rate);

	for(int i=0;i<N;i++){
		BOOST_CHECK_CLOSE(rate[i],h_rate[i],0.01f);
		BOOST_CHECK_CLOSE(W[i],h_W[i],0.01f);
	}
}


BOOST_AUTO_TEST_SUITE_END()
