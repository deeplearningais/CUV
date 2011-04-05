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
#include <fstream>

#include <cuv/tools/cuv_test.hpp>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/dia_matrix.hpp>
#include <cuv/matrix_ops/diagonals.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/basics/sparse_matrix_io.hpp>

using namespace std;
using namespace cuv;

static const int n=32;
static const int m=16;
static const int d=3;
static const int rf=1;

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
	dia_matrix<float,host_memory_space> w;
	Fix()
	:  w(n,m,d,n,rf) 
	{
		std::vector<int> off;
		off.push_back(0);
		off.push_back(1);
		off.push_back(-1);
		w.set_offsets(off);
		sequence(w.vec());
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )

BOOST_AUTO_TEST_CASE( spmv_saveload )
{
	if(1){
		// save...
		std::ofstream ofs("test_dia_mat.save");
		boost::archive::binary_oarchive oa(ofs);
		oa << w;
	}
	dia_matrix<float,host_memory_space> w2;
	if(1){
		// load...
		std::ifstream ifs("test_dia_mat.save");
		boost::archive::binary_iarchive ia(ifs);
		ia >> w2;
	}
	MAT_CMP(w,w2,0.01);
	
}

BOOST_AUTO_TEST_CASE( spmv_uninit )
{
	dia_matrix<float,dev_memory_space> wdev(32,16,3,16,1);
	wdev.dealloc();
	convert(wdev,w);
}


BOOST_AUTO_TEST_CASE( spmv_dia2dense )
{
	// hostdia->hostdense
	dense_matrix<float,host_memory_space,column_major> w2(n,m);
	fill(w2,-1);
	convert(w2,w);
	MAT_CMP(w,w2,0.1);
	//cout << w <<w2;
}

BOOST_AUTO_TEST_CASE( spmv_host2dev )
{
	// host->dev
	dia_matrix<float,dev_memory_space> w2(n,m,w.num_dia(),w.stride(),rf);
	convert(w2,w);
	MAT_CMP(w,w2,0.1);
	fill(w.vec(),0);

	// dev->host
	convert(w,w2);
	MAT_CMP(w,w2,0.1);
}

//BOOST_AUTO_TEST_CASE( avg_dia )
// NOT IMPLEMENTED at the moment
//{
	//cuv::vector<float,host_memory_space> avg( w.num_dia() );
	//avg_diagonals( avg, w );
	//for( int i=0;i<avg.size(); i++ ){
		//BOOST_CHECK_EQUAL( avg[ i ], mean( *w.get_dia( w.get_offset( i ) ) ) );
	//}
//}



BOOST_AUTO_TEST_SUITE_END()
