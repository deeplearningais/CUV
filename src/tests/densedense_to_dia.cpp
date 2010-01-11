#define BOOST_TEST_MODULE densedense_to_dia
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
#include <matrix_ops/densedense_to_sparse.hpp>
#include <timing.hpp>

using namespace std;
using namespace cuv;

static const int n = 64;
static const int m = 32;
static const int k = 16;

struct Fix{
	dev_dia_matrix<float>   C;
	dev_dense_matrix<float> C_;
	dev_dense_matrix<float> A,A_;
	dev_dense_matrix<float> B,B_;
	Fix()
	:   C(n,m,4,max(n,m))
	,   C_(n,m)
	,   A(n,k)
	,   A_(n,k)
	,   B(m,k)
	,   B_(m,k)
	{
		std::vector<int> off;
#if 0
		for(int i=0;i<C.num_dia();i++)
#elif 1
		for(int i=2;i<C.num_dia()+2;i++)
#else
		for(int i=-C.num_dia()/2;i<=C.num_dia()/2;i++)
#endif
			off.push_back(i);
		C.set_offsets(off);
		sequence(A);
		sequence(A_);
		sequence(B);
		sequence(B_);

		sequence(*C.vec());
		host_dia_matrix<float> C2(C.h(),C.w(),C.num_dia(),C.stride());
		host_dense_matrix<float> C_2(C.h(),C.w());
		convert(C2,C);  // dev->host
		convert(C_2,C2); // dia->dense
		convert(C_,C_2); // host->dev
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( dd2s_correctness_dev )
{
	fill(*C.vec(),0);
	sequence(A);
	sequence(B);
	dev_block_descriptor<float> bd(C);
	densedense_to_dia(C,bd,A,B);
	prod(C_,A,B,'n','t');
	//cout << "Dense: "<<endl;
	//for(int i=0;i<C.h();i++){
	//   for(int j=0;j<C.w();j++){
	//           cout << C_(i,j)<<" ";
	//   }
	//   cout <<endl;
	//}
	//cout <<"Dia: "<<endl;
	//for(int i=0;i<C.h();i++){
	//   for(int j=0;j<C.w();j++)
	//           cout << C(i,j)<<" ";
	//   cout <<endl;
	//}
	for(int i=0;i<C.h();i++){
		for(int j=0;j<C.w();j++){
			if(C(i,j) != 0){
				BOOST_CHECK_CLOSE( C(i,j), C_(i,j), 0.01 );
			}
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
