#define BOOST_TEST_MODULE example
#include <iostream>
#include <cstdio>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <cuv_general.hpp>
#include <host_dense_matrix.hpp>
#include <dev_dia_matrix.hpp>
#include <host_dia_matrix.hpp>
#include <convert.hpp>
#include <matrix_ops/matrix_ops.hpp>
#include <timing.hpp>

using namespace std;
using namespace cuv;

static const int n = 8092;
static const int m = 4096;
static const int k = 96;

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

struct Fix{
	host_dia_matrix<float>   A;
	host_dense_matrix<float> A_;
	host_dense_matrix<float> B,B_,BLarge;
	host_dense_matrix<float> C,C_,CLarge;
	Fix()
	:   A(n,m,128,max(n,m))
	,   A_(n,m)
	,   B(m,1)
	,   B_(m,1)
	,   C(n,1)
	,   C_(n,1)
	,   BLarge(m,k)
	,   CLarge(n,k)
	{
		std::vector<int> off;
#if 1
		for(int i=0;i<A.num_dia();i++)
#elif 0
		for(int i=2;i<A.num_dia()+2;i++)
#else
		for(int i=-A.num_dia()/2;i<=A.num_dia()/2;i++)
#endif
			off.push_back(i);
		A.set_offsets(off);
		sequence(A.vec());
		sequence(C);
		sequence(C_);
		sequence(B);
		sequence(B_);
		convert(A_,A);
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( spmv_dev_speed )
{
	dev_dia_matrix<float> A2(n,m,A.num_dia(),A.stride());
	convert(A2,A);
	dev_dense_matrix<float> CLarge2(CLarge.h(), CLarge.w());
	convert(CLarge2,CLarge);
	dev_dense_matrix<float> BLarge2(BLarge.h(), BLarge.w());
	convert(BLarge2,BLarge);

	//float factAv = 2.f, factC = 1.3f;
	float factAv = 1.f, factC = 0.f;
	MEASURE_TIME(host, prod(CLarge, A, BLarge,'n','n',factAv,factC),  10);
	MEASURE_TIME(dev , prod(CLarge2,A2,BLarge2,'n','n',factAv,factC), 10);
	printf("Speedup: %3.4f\n", host/dev);

	MEASURE_TIME(host_t, prod(BLarge,A,CLarge,'t','n',factAv,factC), 10);
	MEASURE_TIME(dev_t , prod(BLarge2,A2,CLarge2,'t','n',factAv,factC), 10);
	printf("Speedup: %3.4f\n", host_t/dev_t);

	BOOST_CHECK_LT(dev,  host);
	BOOST_CHECK_LT(dev_t,host_t);
}
BOOST_AUTO_TEST_CASE( spmv_host_speed )
{
   float factAv = 2.f, factC = 1.3f;
   MEASURE_TIME(sparse, prod(CLarge,A,BLarge,'n','n',factAv,factC), 10);
   MEASURE_TIME(dense , prod(CLarge,A_,BLarge,'n','n',factAv,factC), 10);
   printf("Speedup: %3.4f\n", dense/sparse);

   MEASURE_TIME(sparse_t, prod(BLarge,A,CLarge,'t','n',factAv,factC), 10);
   MEASURE_TIME(dense_t , prod(BLarge,A_,CLarge,'t','n',factAv,factC), 10);
   printf("Speedup: %3.4f\n", dense_t/sparse_t);

   BOOST_CHECK_LT(sparse,  dense);
   BOOST_CHECK_LT(sparse_t,dense_t);
}

BOOST_AUTO_TEST_SUITE_END()
