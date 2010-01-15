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

#define MEASURE_TIME(MSG, OPERATION, ITERS)     \
	float MSG;                                  \
	if(1){                                      \
		Timing tim;                             \
		for(int i=0;i<ITERS;i++){               \
			printf(".");fflush(stdout);         \
			OPERATION ;                         \
		}                                       \
		tim.update(ITERS);                      \
		printf("%s [%s] took %4.4f us/pass\n", #MSG, #OPERATION, 1000000.0f*tim.perf()); \
		MSG = 1000000.0f*tim.perf();            \
	}

using namespace std;
using namespace cuv;

static const int n = 784;
static const int m = 2*784;
static const int k = 96;
static const int fs = 8;
static const int nm = m/n;

struct Fix{
	dev_dia_matrix<float>   C;
	dev_dense_matrix<float> A;
	dev_dense_matrix<float> B;
	Fix()
	:   C(n,m,fs*fs*nm,n)
	,   A(n,k)
	,   B(m,k)
	{
		cerr << "-------------------------------"<<endl;
		std::vector<int> off;
		off.resize(fs*fs*nm);
		for(int i=0;i<fs;i++)
			for(int j=0;j<fs;j++)
				for(int m=0;m<nm;m++)
				{
					off[i*fs+j + m*fs*fs] = i*28+j;
				}
		C.set_offsets(off);
		sequence(A);
		sequence(B);
		sequence(C.vec());
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )

BOOST_AUTO_TEST_CASE( dd2s_speed_host_host )
{
	fill(C.vec(),0);

	host_dia_matrix<float>   C2(C.h(),C.w(),C.num_dia(),C.stride());
	host_dense_matrix<float> C2dense(C.h(),C.w());
	host_dense_matrix<float,column_major> A2(A.h(),A.w());
	host_dense_matrix<float,column_major> B2(B.h(),B.w());
	convert(C2,C);
	convert(A2,A);
	convert(B2,B);
	convert(C2dense,C2);

	host_block_descriptor<float> bdh(C2);
	MEASURE_TIME(host_dense ,prod(C2dense,A2,B2,'n','t'),2);
	MEASURE_TIME(host_dia,   densedense_to_dia(C2,bdh,A2,B2),2);
	printf("Speedup: %3.4f\n", host_dense/host_dia);
}


BOOST_AUTO_TEST_CASE( dd2s_speed_dev_host )
{
	fill(C.vec(),0);

	host_dia_matrix<float> C2(C.h(),C.w(),C.num_dia(),C.stride());
	host_dense_matrix<float,column_major> A2(A.h(),A.w());
	host_dense_matrix<float,column_major> B2(B.h(),B.w());
	convert(C2,C);
	convert(A2,A);
	convert(B2,B);

	dev_block_descriptor<float>  bdd(C);
	host_block_descriptor<float> bdh(C2);
	MEASURE_TIME(dev_dia ,densedense_to_dia(C,bdd,A,B),10);
	MEASURE_TIME(host_dia,densedense_to_dia(C2,bdh,A2,B2),10);
	printf("Speedup: %3.4f\n", host_dia/dev_dia);
}

BOOST_AUTO_TEST_CASE( dd2s_speed_sparse_dense )
{
	if(n>8092 || m>8092)
	   return; // otherwise, we get out of memory errors!
	fill(C.vec(),0);

	// make a dev_dense_matrix equivalent to the dia-matrix
	dev_dense_matrix<float,column_major> Cd(C.h(),C.w());
	host_dia_matrix<float> C2(C.h(),C.w(),C.num_dia(),C.stride());
	host_dense_matrix<float> C_2(C.h(),C.w());
	convert(C2,C);  // dev->host
	convert(C_2,C2); // dia->dense
	convert(Cd,C_2); // host->dev

	dev_block_descriptor<float> bd(C);
	MEASURE_TIME(dev_dia ,densedense_to_dia(C,bd,A,B),10);
	MEASURE_TIME(dev_dense,prod(Cd,A,B,'n','t'),10);
	printf("Speedup: %3.4f\n", dev_dense/dev_dia);
}

BOOST_AUTO_TEST_SUITE_END()

