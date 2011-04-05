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





#define BOOST_TEST_MODULE densedense_to_dia
#include <iostream>
#include <cstdio>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/dense_matrix.hpp>
#include <cuv/basics/dia_matrix.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/matrix_ops/densedense_to_sparse.hpp>
#include <cuv/tools/timing.hpp>

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
	dia_matrix<float,dev_memory_space>   C_dev;
	dense_matrix<float,dev_memory_space,column_major> A_dev;
	dense_matrix<float,dev_memory_space,column_major> B_dev;
	Fix()
	:   C_dev(n,m,fs*fs*nm,n)
	,   A_dev(n,k)
	,   B_dev(m,k)
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
		C_dev.set_offsets(off);
		sequence(A_dev);
		sequence(B_dev);
		sequence(C_dev.vec());
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )

BOOST_AUTO_TEST_CASE( dd2s_speed_host_host )
{
	fill(C_dev.vec(),0);

	dia_matrix<float,host_memory_space>   C2(C_dev.h(),C_dev.w(),C_dev.num_dia(),C_dev.stride());
	dense_matrix<float,host_memory_space,column_major> C2dense(C_dev.h(),C_dev.w());
	dense_matrix<float,host_memory_space,column_major> A2(A_dev.h(),A_dev.w());
	dense_matrix<float,host_memory_space,column_major> B2(B_dev.h(),B_dev.w());
	convert(C2,C_dev);
	convert(A2,A_dev);
	convert(B2,B_dev);
	convert(C2dense,C2);

	host_block_descriptor<float> bdh(C2);
	MEASURE_TIME(host_dense ,prod(C2dense,A2,B2,'n','t'),2);
	MEASURE_TIME(host_dia,   densedense_to_dia(C2,bdh,A2,B2),2);
	printf("Speedup: %3.4f\n", host_dense/host_dia);
}


BOOST_AUTO_TEST_CASE( dd2s_speed_dev_host )
{
	fill(C_dev.vec(),0);

	dia_matrix<float,host_memory_space> C2(C_dev.h(),C_dev.w(),C_dev.num_dia(),C_dev.stride());
	dense_matrix<float,host_memory_space,column_major> A2(A_dev.h(),A_dev.w());
	dense_matrix<float,host_memory_space,column_major> B2(B_dev.h(),B_dev.w());
	convert(C2,C_dev);
	convert(A2,A_dev);
	convert(B2,B_dev);

	dev_block_descriptor<float>  bd_dev(C_dev);
	host_block_descriptor<float> bdh(C2);
	MEASURE_TIME(dev_dia ,densedense_to_dia(C_dev,bd_dev,A_dev,B_dev),10);
	MEASURE_TIME(host_dia,densedense_to_dia(C2,bdh,A2,B2),10);
	printf("Speedup: %3.4f\n", host_dia/dev_dia);
}

BOOST_AUTO_TEST_CASE( dd2s_speed_sparse_dense )
{
	if(n>8092 || m>8092)
	   return; // otherwise, we get out of memory errors!
	fill(C_dev.vec(),0);

	// make a dev_dense_matrix equivalent to the dia-matrix
	dense_matrix<float,dev_memory_space,column_major> Cd(C_dev.h(),C_dev.w());
	dia_matrix<float,host_memory_space> C2(C_dev.h(),C_dev.w(),C_dev.num_dia(),C_dev.stride());
	dense_matrix<float,host_memory_space,column_major> C_2(C_dev.h(),C_dev.w());
	convert(C2,C_dev);  // dev->host
	convert(C_2,C2); // dia->dense
	convert(Cd,C_2); // host->dev

	dev_block_descriptor<float> bd(C_dev);
	MEASURE_TIME(dev_dia ,densedense_to_dia(C_dev,bd,A_dev,B_dev),10);
	MEASURE_TIME(dev_dense,prod(Cd,A_dev,B_dev,'n','t'),10);
	printf("Speedup: %3.4f\n", dev_dense/dev_dia);
}

BOOST_AUTO_TEST_SUITE_END()

