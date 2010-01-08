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

static const int n = 4096;
static const int m = 511;
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
	:   A(n,m,128,min(n,m))
	,   A_(n,m)
	,   B(m,1)
	,   B_(m,1)
	,   C(n,1)
	,   C_(n,1)
	,   BLarge(m,k)
	,   CLarge(n,k)
	{
		std::vector<int> off;
#if 0
		for(int i=0;i<A.num_dia();i++)
#elif 0
		for(int i=2;i<A.num_dia()+2;i++)
#else
		for(int i=-A.num_dia()/2;i<=A.num_dia()/2;i++)
#endif
			off.push_back(i);
		A.set_offsets(off);
		sequence(*A.vec());
		fill(C,0);
		fill(C_,0);
		sequence(B);
		convert(A_,A);
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( spmv_dev_correctness_trans )
{
	sequence(C.vec());
	fill(B.vec(),0);
	dev_dia_matrix<float> A2(n,m,A.num_dia(),A.stride());
	convert(A2,A);
	dev_dense_matrix<float> C2(C.h(),C.w());
	convert(C2,C);
	dev_dense_matrix<float> B2(B.h(),B.w());
	convert(B2,B);

	//for(int i=0;i<A.h();i++){
	//    for(int j=0;j<A.w();j++){
	//        cout << A(i,j) << " ";
	//    }
	//    cout <<endl;
	//}
	prod(B ,A, C, 't','n');
	cout << "Done Initializing..."<<endl;
	prod(B2,A2,C2,'t','n');
	for(int i=0;i<B.vec().size();i++){
		BOOST_CHECK_CLOSE( B.vec()[i], B2.vec()[i], 0.01 );
	}
}
BOOST_AUTO_TEST_CASE( spmv_dev_correctness )
{
  dev_dia_matrix<float> A2(n,m,A.num_dia(),A.stride());
  convert(A2,A);
  dev_dense_matrix<float> C2(C.h(),C.w());
  convert(C2,C);
  dev_dense_matrix<float> B2(B.h(),B.w());
  convert(B2,B);

  prod(C ,A, B, 'n','n');
  prod(C2,A2,B2,'n','n');
  for(int i=0;i<C.vec().size();i++){
	  BOOST_CHECK_CLOSE( C.vec()[i], C2.vec()[i], 1.0 );
  }
}
BOOST_AUTO_TEST_CASE( spmv_dev_speed )
{
  dev_dia_matrix<float> A2(n,m,A.num_dia(),A.stride());
  convert(A2,A);
  dev_dense_matrix<float> CLarge2(CLarge.h(), CLarge.w());
  convert(CLarge2,CLarge);
  dev_dense_matrix<float> BLarge2(BLarge.h(), BLarge.w());
  convert(BLarge2,BLarge);

  MEASURE_TIME(host, prod(CLarge, A, BLarge),  10);
  MEASURE_TIME(dev , prod(CLarge2,A2,BLarge2), 10);
  printf("Speedup: %3.4f\n", host/dev);

  MEASURE_TIME(host_t, prod(BLarge,A,CLarge,'t'), 10);
  MEASURE_TIME(dev_t , prod(BLarge2,A2,CLarge2,'t'), 10);
  printf("Speedup: %3.4f\n", host_t/dev_t);
}
BOOST_AUTO_TEST_CASE( spmv_host_speed )
{
 MEASURE_TIME(sparse, prod(CLarge,A,BLarge), 10);
 MEASURE_TIME(dense , prod(CLarge,A_,BLarge), 10);
 printf("Speedup: %3.4f\n", dense/sparse);

 MEASURE_TIME(sparse_t, prod(BLarge,A,CLarge,'t'), 10);
 MEASURE_TIME(dense_t , prod(BLarge,A_,CLarge,'t'), 10);
 printf("Speedup: %3.4f\n", dense_t/sparse_t);
}
BOOST_AUTO_TEST_CASE( spmv_host_correctness )
{
 prod(C ,A, B,'n','n');
 prod(C_,A_,B,'n','n');
 for(int i=0;i<C.vec().size();i++){
	 BOOST_CHECK_CLOSE( C.vec()[i], C_.vec()[i], 1.0 );
 }
}
BOOST_AUTO_TEST_CASE( spmv_host_correctness_trans )
{
 sequence(C.vec());
 fill(B.vec(), 0);  // reset result for spmv
 fill(B_.vec(),0);  // reset result for dense
 prod(B ,A, C, 't', 'n');
 prod(B_,A_,C, 't', 'n');
 for(int i=0;i<B.vec().size();i++){
	 BOOST_CHECK_CLOSE( B.vec()[i], B_.vec()[i], 1.0 );
 }
}
BOOST_AUTO_TEST_CASE( spmv_host2dev )
{
 // host->dev
 dev_dia_matrix<float> A2(n,m,A.num_dia(),A.stride());
 convert(A2,A);
 for(int i=0;i<A.h();i++){
	 for(int j=0;j<A.w();j++){
		 BOOST_CHECK_CLOSE( A(i,j), A2(i,j), 1.0 );
	 }
 }
 fill(*A.vec(),0);

 // dev->host
 convert(A,A2);
 for(int i=0;i<A.h();i++){
	 for(int j=0;j<A.w();j++){
		 BOOST_CHECK_CLOSE( A(i,j), A2(i,j), 1.0 );
	 }
 }
}



BOOST_AUTO_TEST_SUITE_END()
