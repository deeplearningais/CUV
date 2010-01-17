#define BOOST_TEST_MODULE example
#include <iostream>
#include <cstdio>
#include <cuv_test.hpp>

#include <cuv_general.hpp>
#include <host_dense_matrix.hpp>
#include <dev_dia_matrix.hpp>
#include <host_dia_matrix.hpp>
#include <convert.hpp>
#include <matrix_ops/matrix_ops.hpp>
#include <timing.hpp>

using namespace std;
using namespace cuv;

static const int n = 32;
static const int m = 16;
static const int k = 6;

struct Fix{
	host_dia_matrix<float>   A;
	host_dense_matrix<float> A_;
	host_dense_matrix<float> B,B_;
	host_dense_matrix<float> C,C_;
	Fix()
	:   A(n,m,7,max(n,m),2)
	,   A_(n,m)
	,   B(m,k)
	,   B_(m,k)
	,   C(n,k)
	,   C_(n,k)
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


BOOST_AUTO_TEST_CASE( spmv_dev_correctness_trans )
{
	sequence(C.vec());
	dev_dia_matrix<float> A2(n,m,A.num_dia(),A.stride());
	convert(A2,A);
	dev_dense_matrix<float> C2(C.h(),C.w());
	convert(C2,C);
	dev_dense_matrix<float> B2(B.h(),B.w());
	convert(B2,B);

	prod(B ,A, C, 't','n');
	prod(B2,A2,C2,'t','n');
	MAT_CMP(B,B2,0.1);
}
BOOST_AUTO_TEST_CASE( spmv_dev_correctness )
{
 dev_dia_matrix<float> A2(n,m,A.num_dia(),A.stride());
 convert(A2,A);
 dev_dense_matrix<float> C2(C.h(),C.w());
 convert(C2,C);
 dev_dense_matrix<float> B2(B.h(),B.w());
 convert(B2,B);

 float factAv = 2.f, factC = 1.3f;
 prod(C ,A, B, 'n','n', factAv, factC);
 prod(C2,A2,B2,'n','n', factAv, factC);
 MAT_CMP(C,C2,0.1);
}
BOOST_AUTO_TEST_CASE( spmv_host_correctness )
{
	float factAv = 2.f, factC = 1.3f;
	prod(C ,A, B,'n','n',factAv,factC);
	prod(C_,A_,B,'n','n',factAv,factC);
	MAT_CMP(C,C_,0.1);
}
BOOST_AUTO_TEST_CASE( spmv_host_correctness_trans )
{
	float factAv = 2.f, factC = 1.3f;
	prod(B ,A, C, 't', 'n',factAv,factC);
	prod(B_,A_,C, 't', 'n',factAv,factC);
	MAT_CMP(B,B_,0.5);
}



BOOST_AUTO_TEST_SUITE_END()
