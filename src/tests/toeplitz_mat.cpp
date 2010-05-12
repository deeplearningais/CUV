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
#include <boost/format.hpp>

#include <tools/cuv_test.hpp>
#include <tools/cuv_general.hpp>
#include <vector_ops/vector_ops.hpp>
#include <basics/toeplitz_matrix.hpp>
#include <convert/convert.hpp>
#include <basics/sparse_matrix_io.hpp>
#include <basics/filter_factory.hpp>

using namespace std;
using namespace cuv;

static const int n=64;
static const int m=64;
static const int ms=4;
static const int d=3*ms;


struct Fix{
	toeplitz_matrix<float,host_memory_space> w;
	Fix()
	:  w(n,m,d,ms,ms) 
	{
		std::vector<int> off;
		for( int k=-2;k<ms-2;k++ ){
			off.push_back( k*m/ms-2);
			off.push_back( k*m/ms-0);
			off.push_back( k*m/ms+2);
		}
		w.set_offsets(off);
		sequence(w.vec());
		apply_scalar_functor( w.vec(), SF_ADD, 1.f );
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )

BOOST_AUTO_TEST_CASE( output )
{
	typedef toeplitz_matrix<float, host_memory_space> sparse_type;
	//typedef dia_matrix<float, host_memory_space> sparse_type;

	filter_factory<float, host_memory_space>   ff(8,8,3,4,4);
	dense_matrix<float,column_major,host_memory_space> filters(3*3, 4*4);
	fill( filters.vec(), 0.f );
	for( int i=0; i<4*4;i++ ){
		for (int j = 0; j < 3*3; ++j) 
		{
			filters.set( j,i , ( float )i+1);
		}
	}
	cout << "filters: "<< filters << endl;
	sparse_type* mat_p = ff.create_toeplitz_from_filters(filters);
	//sparse_type* mat_p = ff.get_dia();
	sparse_type& mat = *mat_p;
	BOOST_CHECK_EQUAL( 8*8*4, mat.w() )	;
	BOOST_CHECK_EQUAL( 8*8*4, mat.h() )	;
	//sequence( mat.vec() );
	//apply_scalar_functor( mat.vec(), SF_ADD, 1.f );

	using boost::format;
	cout <<"   | ";
	for( int j=0; j< mat.w(); j++ )
		cout << ( format( "%03d " )%j ).str();
	cout <<endl;
	for( int i=0; i< mat.w(); i++ ){
		cout << ( format( "%03d | " )%i ).str();
		for( int j=0; j< mat.h(); j++ ){
			if( mat( i,j ) == 0 )
				cout << "    ";
			else
				cout << ( format( "%03d " ) % ( mat( i,j ) ) ).str();
		}
		cout <<endl;
	}
	cout << *ff.extract_filters( mat )<<endl;
	delete mat_p;
}

BOOST_AUTO_TEST_CASE( spmv_uninit )
{
	toeplitz_matrix<float,dev_memory_space> wdev(32,16,3);
	wdev.dealloc();
	convert(wdev,w);
	MAT_CMP( wdev, w , 0.01);
}


BOOST_AUTO_TEST_CASE( spmv_dia2dense )
{
	// hostdia->hostdense
	dense_matrix<float,column_major,host_memory_space> w2(n,m);
	fill(w2.vec(),-1);
	convert(w2,w);
	MAT_CMP(w,w2,0.1);
}

BOOST_AUTO_TEST_CASE( spmv_host2dev )
{
	// host->dev
	toeplitz_matrix<float,dev_memory_space> w2(n,m,w.num_dia());
	convert(w2,w);
	MAT_CMP(w,w2,0.1);
	fill(w.vec(),0);

	// dev->host
	convert(w,w2);
	MAT_CMP(w,w2,0.1);
}



BOOST_AUTO_TEST_SUITE_END()
