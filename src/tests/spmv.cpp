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

static const int n = 32;
static const int m = 16;
static const int k = 2;

struct Fix{
	host_dia_matrix<float>   A;
	host_dense_matrix<float> A_;
	host_dense_matrix<float> B,B_;
	host_dense_matrix<float> C,C_;
	Fix()
	:   A(n,m,7,max(n,m))
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
		sequence(*A.vec());
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

 float factAv = 2.f, factC = 1.3f;
 prod(C ,A, B, 'n','n', factAv, factC);
 prod(C2,A2,B2,'n','n', factAv, factC);
 for(int i=0;i<C.vec().size();i++){
	 BOOST_CHECK_CLOSE( C.vec()[i], C2.vec()[i], 1.0 );
 }
}
BOOST_AUTO_TEST_CASE( spmv_host_correctness )
{
	float factAv = 2.f, factC = 1.3f;
	prod(C ,A, B,'n','n',factAv,factC);
	prod(C_,A_,B,'n','n',factAv,factC);
	for(int i=0;i<C.vec().size();i++){
		BOOST_CHECK_CLOSE( C.vec()[i], C_.vec()[i], 1.0 );
	}
}
BOOST_AUTO_TEST_CASE( spmv_host_correctness_trans )
{
	float factAv = 2.f, factC = 1.3f;
	prod(B ,A, C, 't', 'n',factAv,factC);
	prod(B_,A_,C, 't', 'n',factAv,factC);
	for(int i=0;i<B.vec().size();i++){
		BOOST_CHECK_CLOSE( B.vec()[i], B_.vec()[i], 1.0 );
	}
}



BOOST_AUTO_TEST_SUITE_END()
