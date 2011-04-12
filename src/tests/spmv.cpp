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
#include <cuv/tools/cuv_test.hpp>

#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/tensor.hpp>
#include <cuv/basics/dia_matrix.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/tools/timing.hpp>

using namespace std;
using namespace cuv;

static const int n = 32;
static const int m = 16;
static const int k = 6;
static const int rf = 1;

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
	dia_matrix<float,host_memory_space>   A;
	tensor<float,host_memory_space,column_major> A_;
	tensor<float,host_memory_space,column_major> B,B_;
	tensor<float,host_memory_space,column_major> C,C_;
	Fix()
	:   A(n,m,7,max(n,m),rf)
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
	sequence(C);
	dia_matrix<float,dev_memory_space> A2(n,m,A.num_dia(),A.stride(),rf);
	convert(A2,A);
	tensor<float,dev_memory_space,column_major> C2(C.shape());
	convert(C2,C);
	tensor<float,dev_memory_space,column_major> B2(B.shape());
	convert(B2,B);
	prod(B ,A, C, 't','n');
	prod(B2,A2,C2,'t','n');
	MAT_CMP(B,B2,0.1);
}
BOOST_AUTO_TEST_CASE( spmv_dev_correctness )
{
 dia_matrix<float,dev_memory_space> A2(n,m,A.num_dia(),A.stride(),rf);
 convert(A2,A);
 tensor<float,dev_memory_space,column_major> C2(C.shape());
 convert(C2,C);
 tensor<float,dev_memory_space,column_major> B2(B.shape());
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
