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

#include <cuv/tools/cuv_test.hpp>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/tensor.hpp>
#include <cuv/basics/dia_matrix.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/matrix_ops/densedense_to_sparse.hpp>
#include <cuv/tools/timing.hpp>

using namespace std;
using namespace cuv;

static const int n = 36;
static const int m = 19;
static const int k = 16;
static const int rf = 4;

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
	dia_matrix<float,dev_memory_space>   C;
	tensor<float,dev_memory_space,column_major> C_;
	tensor<float,dev_memory_space,column_major> A,A_;
	tensor<float,dev_memory_space,column_major> B,B_;
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
		dia_matrix<float,host_memory_space> C2(C.shape()[0],C.shape()[1],C.num_dia(),C.stride(),rf);
		tensor<float,host_memory_space,column_major> C_2(C.shape()[0],C.shape()[1]);
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
	//for(int i=0;i<C.shape()[0];i++){
	//   for(int j=0;j<C.shape()[1];j++){
	//           cout << C_(i,j)<<" ";
	//   }
	//   cout <<endl;
	//}
	//cout <<"Dia: "<<endl;
	//for(int i=0;i<C.shape()[0];i++){
	//   for(int j=0;j<C.shape()[1];j++)
	//           cout << C(i,j)<<" ";
	//   cout <<endl;
	//}
	for(int i=0;i<C.shape()[0];i++){
		for(int j=0;j<C.shape()[1];j++){
			if(C(i,j) != 0){
				BOOST_CHECK_CLOSE( (float)C(i,j), (float)C_(i,j), 0.01 );
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( dd2s_correctness_host )
{
	fill(C.vec(),0);
	sequence(A);
	sequence(B);

	tensor<float,host_memory_space,column_major> Chdense(C.shape()[0],C.shape()[1]);
	dia_matrix<float,host_memory_space>   Chdia(C.shape()[0],C.shape()[1],C.num_dia(),C.stride(),rf);
	tensor<float,host_memory_space,column_major> Ah(A.shape());
	tensor<float,host_memory_space,column_major> Bh(B.shape());
	convert(Chdia,C);
	convert(Chdense,Chdia);
	convert(Ah,A);
	convert(Bh,B);

	host_block_descriptor<float> bdh(Chdia);
	densedense_to_dia(Chdia,bdh,Ah,Bh);
	prod(Chdense,Ah,Bh,'n','t');
	for(int i=0;i<C.shape()[0];i++){
		for(int j=0;j<C.shape()[1];j++){
			if(Chdia(i,j) != 0){
				BOOST_CHECK_CLOSE( (float)Chdia(i,j), (float)Chdense(i,j), 0.01 );
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( dd2s_cmp_dev_host )
{
	fill(C.vec(),0);
	sequence(A);
	sequence(B);

	dia_matrix<float,host_memory_space> C2(C.shape()[0],C.shape()[1],C.num_dia(),C.stride(),rf);
	tensor<float,host_memory_space,column_major> A2(A.shape());
	tensor<float,host_memory_space,column_major> B2(B.shape());
	convert(C2,C);
	convert(A2,A);
	convert(B2,B);

	dev_block_descriptor<float>  bd(C);
	host_block_descriptor<float> bdh(C2);
	densedense_to_dia(C,bd,A,B);
	densedense_to_dia(C2,bdh,A2,B2);
	
	//cout <<"Host: "<<endl;
	//for(int i=0;i<C2.shape()[0];i++){
	//  for(int j=0;j<C2.shape()[1];j++)
	//          cout << C2(i,j)<<" ";
	//  cout <<endl;
	//}
	//cout <<"Dev: "<<endl;
	//for(int i=0;i<C.shape()[0];i++){
	//  for(int j=0;j<C.shape()[1];j++)
	//          cout << C(i,j)<<" ";
	//  cout <<endl;
	//}
	MAT_CMP(C,C2,0.01);
}

BOOST_AUTO_TEST_SUITE_END()
