//*LB*
// Copyright (c) 2010, University of Bonn, Institute for Computer Science VI
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of the University of Bonn 
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*LE*





#define BOOST_TEST_MODULE example
#include <iostream>
#include <cstdio>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <cuv_general.hpp>
#include <dense_matrix.hpp>
#include <basics/toeplitz_matrix.hpp>
#include <basics/filter_factory.hpp>
#include <convert.hpp>
#include <matrix_ops/matrix_ops.hpp>
#include <matrix_ops/diagonals.hpp>
#include <timing.hpp>

using namespace std;
using namespace cuv;

static const unsigned int px = 128;  // image width and height
static const unsigned int im = 1;
static const unsigned int om = 4;
static const unsigned int  n = im*px*px;// size of input layer
static const unsigned int  m = om*px*px;  // size output layer (same as input times number of output maps)
static const unsigned int  k = 14;    // number images
static const unsigned int fs = 11;    // filter size

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

#define NO_DENSE_TESTS 1

struct Fix{
	toeplitz_matrix<float,host_memory_space>   A;
	dense_matrix<float,column_major,host_memory_space> A_;
	dense_matrix<float,column_major,host_memory_space> B,B_,BLarge;
	dense_matrix<float,column_major,host_memory_space> C,C_,CLarge;
	Fix()
	:   A()
	,   A_(n,m)
	,   B(m,1)
	,   B_(m,1)
	,   C(n,1)
	,   C_(n,1)
	,   BLarge(m,k)
	,   CLarge(n,k)
	{
		std::vector<int> off;
		A = *filter_factory<float,host_memory_space>(px,px,fs,im,om).get_toeplitz();
		sequence(A.vec());
		sequence(C);
		sequence(B);
		if(NO_DENSE_TESTS)
			return;
		sequence(B_);
		sequence(C_);
		convert(A_,A);
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( spmv_dev_speed_vs_dense )
{
	if(NO_DENSE_TESTS)
		return;
	dense_matrix<float,column_major,host_memory_space> Ahostdense(n,m);
	convert(Ahostdense,A);

	dense_matrix<float,column_major,dev_memory_space> Adevdense(n,m);
	convert(Adevdense,Ahostdense);

	toeplitz_matrix<float,dev_memory_space>   Adevdia(n,m,A.num_dia());
	convert(Adevdia,A);

	dense_matrix<float,column_major,dev_memory_space> CLarge2(CLarge.h(), CLarge.w());
	convert(CLarge2,CLarge);
	dense_matrix<float,column_major,dev_memory_space> BLarge2(BLarge.h(), BLarge.w());
	convert(BLarge2,BLarge);

	float factAv = 2.f, factC = 1.3f;
	//float factAv = 1.f, factC = 0.f;
	MEASURE_TIME(dev_dense, prod(CLarge2, Adevdense, BLarge2,'n','n',factAv,factC),  10);
	MEASURE_TIME(dev_dia , prod(CLarge2,Adevdia,BLarge2,'n','n',factAv,factC), 10);
	printf("Speedup: %3.4f\n", dev_dense/dev_dia);

	MEASURE_TIME(dev_dense_t, prod(BLarge2,Adevdense,CLarge2,'t','n',factAv,factC), 10);
	MEASURE_TIME(dev_dia_t , prod(BLarge2,Adevdense,CLarge2,'t','n',factAv,factC), 10);
	printf("Speedup: %3.4f\n", dev_dense_t/dev_dia_t);

	BOOST_CHECK_LT(dev_dia,  dev_dense);
	BOOST_CHECK_LT(dev_dia_t,dev_dense_t);
}

BOOST_AUTO_TEST_CASE( avg_diagonals_speed )
{
	typedef toeplitz_matrix<float, dev_memory_space> toeplitz_t;
	typedef dia_matrix<float, dev_memory_space> dia_t;

	typedef toeplitz_matrix<float, host_memory_space> toeplitz_ht;
	typedef dia_matrix<float, host_memory_space> dia_ht;

	filter_factory<float, dev_memory_space>   ff(px,px,fs,im,om);

	auto_ptr<dia_t>     mat_p(ff.get_dia());
	auto_ptr<toeplitz_t> tp_p(ff.get_toeplitz());
	dia_t&      mat   = *mat_p;
	toeplitz_t& tp    = *tp_p;
	sequence( mat.vec());
	fill( tp.vec(), -1 );

	toeplitz_ht tph;
	dia_ht      math;
	convert( tph,  tp );
	convert( math, mat );

   MEASURE_TIME(avg_dia_host, avg_diagonals( tph,math ), 10);
   MEASURE_TIME(avg_dia_dev , avg_diagonals( tp,mat ), 10);
   printf("Speedup: %3.4f\n", avg_dia_host/avg_dia_dev);
}
BOOST_AUTO_TEST_CASE( spmv_dev_speed_vs_toeplitz )
{
	toeplitz_matrix<float,dev_memory_space> A2;
	convert(A2,A);
	dense_matrix<float,column_major,dev_memory_space> CLarge2(CLarge.h(), CLarge.w());
	convert(CLarge2,CLarge);
	dense_matrix<float,column_major,dev_memory_space> BLarge2(BLarge.h(), BLarge.w());
	convert(BLarge2,BLarge);

	float factAv = 2.f, factC = 1.3f;
	//float factAv = 1.f, factC = 0.f;
	MEASURE_TIME(host_dia, prod(CLarge, A, BLarge,'n','n',factAv,factC),  2);
	MEASURE_TIME(dev_dia , prod(CLarge2,A2,BLarge2,'n','n',factAv,factC), 2);
	printf("Speedup: %3.4f\n", host_dia/dev_dia);

	MEASURE_TIME(host_dia_t, prod(BLarge,A,CLarge,'t','n',factAv,factC), 2);
	MEASURE_TIME(dev_dia_t , prod(BLarge2,A2,CLarge2,'t','n',factAv,factC), 2);
	printf("Speedup: %3.4f\n", host_dia_t/dev_dia_t);

	BOOST_CHECK_LT(dev_dia,  host_dia);
	BOOST_CHECK_LT(dev_dia_t,host_dia_t);
}
BOOST_AUTO_TEST_CASE( spmv_host_speed )
{
	if(NO_DENSE_TESTS)
		return;
   float factAv = 2.f, factC = 1.3f;
   MEASURE_TIME(sparse_host, prod(CLarge,A,BLarge,'n','n',factAv,factC), 10);
   MEASURE_TIME(dense_host , prod(CLarge,A_,BLarge,'n','n',factAv,factC), 10);
   printf("Speedup: %3.4f\n", dense_host/sparse_host);

   MEASURE_TIME(sparse_host_t, prod(BLarge,A,CLarge,'t','n',factAv,factC), 10);
   MEASURE_TIME(dense_host_t , prod(BLarge,A_,CLarge,'t','n',factAv,factC), 10);
   printf("Speedup: %3.4f\n", dense_host_t/sparse_host_t);

   BOOST_CHECK_LT(sparse_host,  dense_host);
   BOOST_CHECK_LT(sparse_host_t,dense_host_t);
}

BOOST_AUTO_TEST_SUITE_END()
