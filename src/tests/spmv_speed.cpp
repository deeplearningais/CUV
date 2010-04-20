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
#include <dia_matrix.hpp>
#include <convert.hpp>
#include <matrix_ops/matrix_ops.hpp>
#include <timing.hpp>

using namespace std;
using namespace cuv;

static const int  n = 150*150;
static const int  m = 150*150;
static const int  k = 96;
static const int rf = 2;
static const int fs = 8;
static const int px = 150;
static const int nm = m/n;

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
	dia_matrix<float,host_memory_space>   A;
	dense_matrix<float,column_major,host_memory_space> A_;
	dense_matrix<float,column_major,host_memory_space> B,B_,BLarge;
	dense_matrix<float,column_major,host_memory_space> C,C_,CLarge;
	Fix()
	:   A(n,m,fs*fs*nm,n,rf)
	,   A_(n,m)
	,   B(m,1)
	,   B_(m,1)
	,   C(n,1)
	,   C_(n,1)
	,   BLarge(m,k)
	,   CLarge(n,k)
	{
		std::vector<int> off;
		off.resize(fs*fs*nm);
		for(int i=0;i<fs;i++)
			for(int j=0;j<fs;j++)
				for(int m=0;m<nm;m++)
				{
					off[i*fs+j + m*fs*fs] = i*px+j;
				}
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


BOOST_AUTO_TEST_CASE( spmv_dev_speed_vs_dense )
{
	if(n*m > 1000*1000)
		return;
	dense_matrix<float,column_major,host_memory_space> Ahostdense(n,m);
	convert(Ahostdense,A);

	dense_matrix<float,column_major,dev_memory_space> Adevdense(n,m);
	convert(Adevdense,Ahostdense);

	dia_matrix<float,dev_memory_space>   Adevdia(n,m,A.num_dia(),A.stride(),rf);
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
BOOST_AUTO_TEST_CASE( spmv_dev_speed_vs_dia )
{
	dia_matrix<float,dev_memory_space> A2(n,m,A.num_dia(),A.stride(),rf);
	convert(A2,A);
	dense_matrix<float,column_major,dev_memory_space> CLarge2(CLarge.h(), CLarge.w());
	convert(CLarge2,CLarge);
	dense_matrix<float,column_major,dev_memory_space> BLarge2(BLarge.h(), BLarge.w());
	convert(BLarge2,BLarge);

	float factAv = 2.f, factC = 1.3f;
	//float factAv = 1.f, factC = 0.f;
	MEASURE_TIME(host_dia, prod(CLarge, A, BLarge,'n','n',factAv,factC),  10);
	MEASURE_TIME(dev_dia , prod(CLarge2,A2,BLarge2,'n','n',factAv,factC), 10);
	printf("Speedup: %3.4f\n", host_dia/dev_dia);

	MEASURE_TIME(host_dia_t, prod(BLarge,A,CLarge,'t','n',factAv,factC), 10);
	MEASURE_TIME(dev_dia_t , prod(BLarge2,A2,CLarge2,'t','n',factAv,factC), 10);
	printf("Speedup: %3.4f\n", host_dia_t/dev_dia_t);

	BOOST_CHECK_LT(dev_dia,  host_dia);
	BOOST_CHECK_LT(dev_dia_t,host_dia_t);
}
BOOST_AUTO_TEST_CASE( spmv_host_speed )
{
	if(n*m > 1000*1000)
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
