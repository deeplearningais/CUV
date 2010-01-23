#define BOOST_TEST_MODULE densedense_to_dia
#include <iostream>
#include <cstdio>

#include <cuv_test.hpp>
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

static const int n = 36;
static const int m = 19;
static const int k = 16;
static const int rf = 4;

struct Fix{
	dev_dia_matrix<float>   C;
	dev_dense_matrix<float> C_;
	dev_dense_matrix<float> A,A_;
	dev_dense_matrix<float> B,B_;
	Fix()
	:   C(n,m,7,max(n,m),rf)
	,   C_(n,m)
	,   A(n,k)
	,   A_(n,k)
	,   B(m,k)
	,   B_(m,k)
	{
		std::vector<int> off;
#if 0
		for(int i=0;i<C.num_dia();i++)
#elif 0
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

		sequence(C.vec());
		host_dia_matrix<float> C2(C.h(),C.w(),C.num_dia(),C.stride(),rf);
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
	fill(C.vec(),0);
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

BOOST_AUTO_TEST_CASE( dd2s_correctness_host )
{
	fill(C.vec(),0);
	sequence(A);
	sequence(B);

	host_dense_matrix<float> Chdense(C.h(),C.w());
	host_dia_matrix<float>   Chdia(C.h(),C.w(),C.num_dia(),C.stride(),rf);
	host_dense_matrix<float,column_major> Ah(A.h(),A.w());
	host_dense_matrix<float,column_major> Bh(B.h(),B.w());
	convert(Chdia,C);
	convert(Chdense,Chdia);
	convert(Ah,A);
	convert(Bh,B);

	host_block_descriptor<float> bdh(Chdia);
	densedense_to_dia(Chdia,bdh,Ah,Bh);
	prod(Chdense,Ah,Bh,'n','t');
	for(int i=0;i<C.h();i++){
		for(int j=0;j<C.w();j++){
			if(Chdia(i,j) != 0){
				BOOST_CHECK_CLOSE( Chdia(i,j), Chdense(i,j), 0.01 );
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( dd2s_cmp_dev_host )
{
	fill(C.vec(),0);
	sequence(A);
	sequence(B);

	host_dia_matrix<float> C2(C.h(),C.w(),C.num_dia(),C.stride(),rf);
	host_dense_matrix<float,column_major> A2(A.h(),A.w());
	host_dense_matrix<float,column_major> B2(B.h(),B.w());
	convert(C2,C);
	convert(A2,A);
	convert(B2,B);

	dev_block_descriptor<float>  bd(C);
	host_block_descriptor<float> bdh(C2);
	densedense_to_dia(C,bd,A,B);
	densedense_to_dia(C2,bdh,A2,B2);
	
	//cout <<"Host: "<<endl;
	//for(int i=0;i<C2.h();i++){
	//  for(int j=0;j<C2.w();j++)
	//          cout << C2(i,j)<<" ";
	//  cout <<endl;
	//}
	//cout <<"Dev: "<<endl;
	//for(int i=0;i<C.h();i++){
	//  for(int j=0;j<C.w();j++)
	//          cout << C(i,j)<<" ";
	//  cout <<endl;
	//}
	MAT_CMP(C,C2,0.01);
}

BOOST_AUTO_TEST_SUITE_END()
