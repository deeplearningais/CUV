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

#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/dense_matrix.hpp>
#include <cuv/basics/dia_matrix.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/tools/timing.hpp>

using namespace std;
using namespace cuv;

static const unsigned int px = 256;  // image width and height
static const unsigned int  n = px*px;// size of input layer
static const unsigned int  m = 16*n;  // size output layer (same as input times number of output maps)
static const unsigned int  k = 14;    // number images
static const unsigned int fs = 8;    // filter size
static const unsigned int nm = m/n;  // number of maps in output layer

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

struct MyConfig {
	static const int dev = CUDA_TEST_DEVICE;
	MyConfig()   { 
		printf("Testing on device=%d\n",dev);
		initCUDA(dev); 
	}
	~MyConfig()  { exitCUDA();  }
};

BOOST_GLOBAL_FIXTURE( MyConfig );

struct Fix{
	dia_matrix<float,host_memory_space>   A_host;
	dense_matrix<float,column_major,host_memory_space> A_;
	dense_matrix<float,column_major,host_memory_space> B,B_,BLarge_host;
	dense_matrix<float,column_major,host_memory_space> C,C_,CLarge_host;
	Fix()
	:   A_host(n,m,fs*fs*nm,n)
	,   A_(n,m)
	,   B(m,1)
	,   B_(m,1)
	,   C(n,1)
	,   C_(n,1)
	,   BLarge_host(m,k)
	,   CLarge_host(n,k)
	{
		std::vector<int> off;
		off.resize(fs*fs*nm);
		for(int i=0;i<fs;i++)
			for(int j=0;j<fs;j++)
				for(int m=0;m<nm;m++)
				{
					off[i*fs+j + m*fs*fs] = i*px+j;
				}
		A_host.set_offsets(off);
		sequence(A_host.vec());
		sequence(C);
		sequence(B);
		if(px>64)
			return;
		sequence(B_);
		sequence(C_);
		convert(A_,A_host);
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( spmv_dev_speed_vs_dense )
{
	if(px>64)
		return;
	dense_matrix<float,column_major,host_memory_space> Ahostdense(n,m);
	convert(Ahostdense,A_host);

	dense_matrix<float,column_major,dev_memory_space> Adevdense(n,m);
	convert(Adevdense,Ahostdense);

	dia_matrix<float,dev_memory_space>   Adevdia(n,m,A_host.num_dia(),A_host.stride());
	convert(Adevdia,A_host);

	dense_matrix<float,column_major,dev_memory_space> CLarge2_dev(CLarge_host.h(), CLarge_host.w());
	convert(CLarge2_dev,CLarge_host);
	dense_matrix<float,column_major,dev_memory_space> BLarge2(BLarge_host.h(), BLarge_host.w());
	convert(BLarge2,BLarge_host);

	float factAv = 2.f, factC = 1.3f;
	//float factAv = 1.f, factC = 0.f;
	MEASURE_TIME(dev_dense, prod(CLarge2_dev, Adevdense, BLarge2,'n','n',factAv,factC),  10);
	MEASURE_TIME(dev_dia , prod(CLarge2_dev,Adevdia,BLarge2,'n','n',factAv,factC), 10);
	printf("Speedup: %3.4f\n", dev_dense/dev_dia);

	MEASURE_TIME(dev_dense_t, prod(BLarge2,Adevdense,CLarge2_dev,'t','n',factAv,factC), 10);
	MEASURE_TIME(dev_dia_t , prod(BLarge2,Adevdense,CLarge2_dev,'t','n',factAv,factC), 10);
	printf("Speedup: %3.4f\n", dev_dense_t/dev_dia_t);

	BOOST_CHECK_LT(dev_dia,  dev_dense);
	BOOST_CHECK_LT(dev_dia_t,dev_dense_t);
}
BOOST_AUTO_TEST_CASE( spmv_dev_speed_vs_dia )
{
	dia_matrix<float,dev_memory_space> A2(n,m,A_host.num_dia(),A_host.stride());
	convert(A2,A_host);
	dense_matrix<float,column_major,dev_memory_space> CLarge2_dev(CLarge_host.h(), CLarge_host.w());
	convert(CLarge2_dev,CLarge_host);
	dense_matrix<float,column_major,dev_memory_space> BLarge2(BLarge_host.h(), BLarge_host.w());
	convert(BLarge2,BLarge_host);

	float factAv = 2.f, factC = 1.3f;
	//float factAv = 1.f, factC = 0.f;
	MEASURE_TIME(host_dia, prod(CLarge_host, A_host, BLarge_host,'n','n',factAv,factC),  2);
	MEASURE_TIME(dev_dia , prod(CLarge2_dev,A2,BLarge2,'n','n',factAv,factC), 2);
	printf("Speedup: %3.4f\n", host_dia/dev_dia);

	MEASURE_TIME(host_dia_t, prod(BLarge_host,A_host,CLarge_host,'t','n',factAv,factC), 2);
	MEASURE_TIME(dev_dia_t , prod(BLarge2,A2,CLarge2_dev,'t','n',factAv,factC), 2);
	printf("Speedup: %3.4f\n", host_dia_t/dev_dia_t);

	BOOST_CHECK_LT(dev_dia,  host_dia);
	BOOST_CHECK_LT(dev_dia_t,host_dia_t);
}
BOOST_AUTO_TEST_CASE( spmv_host_speed )
{
	if(px>64)
		return;
   float factAv = 2.f, factC = 1.3f;
   MEASURE_TIME(sparse_host, prod(CLarge_host,A_host,BLarge_host,'n','n',factAv,factC), 10);
   MEASURE_TIME(dense_host , prod(CLarge_host,A_,BLarge_host,'n','n',factAv,factC), 10);
   printf("Speedup: %3.4f\n", dense_host/sparse_host);

   MEASURE_TIME(sparse_host_t, prod(BLarge_host,A_host,CLarge_host,'t','n',factAv,factC), 10);
   MEASURE_TIME(dense_host_t , prod(BLarge_host,A_,CLarge_host,'t','n',factAv,factC), 10);
   printf("Speedup: %3.4f\n", dense_host_t/sparse_host_t);

   BOOST_CHECK_LT(sparse_host,  dense_host);
   BOOST_CHECK_LT(sparse_host_t,dense_host_t);
}

BOOST_AUTO_TEST_SUITE_END()
