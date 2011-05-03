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
#include <boost/test/included/unit_test.hpp>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/tensor.hpp>
#include <cuv/basics/memory2d.hpp>
using namespace cuv;

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
	Fix()
	{
	}
	~Fix(){
	}
};

BOOST_FIXTURE_TEST_SUITE( s, Fix )

/** 
 * @test
 * @brief create tensor
 */
BOOST_AUTO_TEST_CASE( create_tensor )
{
	// column_major
	tensor<float,host_memory_space,column_major> m(extents[2][3][4]);
	BOOST_CHECK_EQUAL(24,m.size());
	BOOST_CHECK_EQUAL(2ul,m.shape()[0]);
	BOOST_CHECK_EQUAL(3ul,m.shape()[1]);
	BOOST_CHECK_EQUAL(4ul,m.shape()[2]);

	BOOST_CHECK_EQUAL(0ul,m.index_of(extents[0][0][0]));  // column major test
	BOOST_CHECK_EQUAL(1ul,m.index_of(extents[1][0][0]));
	BOOST_CHECK_EQUAL(2ul,m.index_of(extents[0][1][0]));


	// row_major
	tensor<float,host_memory_space,row_major> n(extents[2][3][4]);
	BOOST_CHECK_EQUAL(24,m.size());
	BOOST_CHECK_EQUAL(2ul,n.shape()[0]);
	BOOST_CHECK_EQUAL(3ul,n.shape()[1]);
	BOOST_CHECK_EQUAL(4ul,n.shape()[2]);

	BOOST_CHECK_EQUAL(0ul,n.index_of(extents[0][0][0]));  // row major test
	BOOST_CHECK_EQUAL(1ul,n.index_of(extents[0][0][1]));
	BOOST_CHECK_EQUAL(2ul,n.index_of(extents[0][0][2]));
	BOOST_CHECK_EQUAL(4ul,n.index_of(extents[0][1][0]));
}

BOOST_AUTO_TEST_CASE( tensor_data_access )
{
	tensor<float,host_memory_space,column_major> m(extents[2][3][4]);
	tensor<float,host_memory_space,row_major>    n(extents[2][3][4]);

	tensor<float,host_memory_space,column_major> o(extents[2][3][4]);
	tensor<float,host_memory_space,row_major>    p(extents[2][3][4]);
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 3; ++j) {
			for (int k = 0; k < 4; ++k) {
				m(i,j,k)=i*j+k;
				n(i,j,k)=i*j+k;

				o(i,j,k)=i*j+k;
				p(i,j,k)=i*j+k;
			}
		}
	}
	BOOST_CHECK_EQUAL(1*2+3,m(1,2,3));
	BOOST_CHECK_EQUAL(1*2+3,n(1,2,3));
	BOOST_CHECK_EQUAL(1*2+3,o(1,2,3));
	BOOST_CHECK_EQUAL(1*2+3,p(1,2,3));

	BOOST_CHECK_EQUAL(1*2+3-1,--p(1,2,3));
	BOOST_CHECK_EQUAL(1*2+3,  p(1,2,3)+=1);
}

BOOST_AUTO_TEST_CASE( tensor_assignment )
{
	tensor<float,host_memory_space,column_major> m(extents[2][3][4]);
	tensor<float,host_memory_space,column_major> n(extents[2][3][4]);

	tensor<float,host_memory_space,column_major> o(extents[2][3][4]);

	for (int i = 0; i < 2*3*4; ++i)
		m[i] = i;
	n = m;
	o = m;

	tensor<float,host_memory_space,column_major> s(n);
	tensor<float,dev_memory_space,column_major> t(n);

	for (int i = 0; i < 2*3*4; ++i){
		BOOST_CHECK_EQUAL(m[i], i);
		BOOST_CHECK_EQUAL(n[i], i);
		BOOST_CHECK_EQUAL(o[i], i);
		BOOST_CHECK_EQUAL(s[i], i);
		BOOST_CHECK_EQUAL(t[i], i);
	}


}

template<class V, class M>
void test_mem2d(){
	typedef memory2d<V,M,V*> mem_t;

	mem_t l(120,240);
	mem_t k(120,240, l.ptr(),true);
	BOOST_CHECK_EQUAL(l.ptr(), k.ptr());

#define P(X) #X<<":"<<(X)<<"  "
	tensor<V,M,row_major>       t1d(extents[123][247]);
	tensor<V,M,row_major,mem_t> t2d(extents[123][247]);
	if(IsSame<M,dev_memory_space>::Result::value){
		// for this test to be useful, we would like to see a pitched memory at least on device!
		bool has_pitch = t2d.data().width()*sizeof(V) != t2d.data().pitch();
		BOOST_CHECK(has_pitch);
	}
	
	for(int i=0;i<120; i++)
		for(int j=0;j<240; j++){
			t1d(i,j) = (V) 0;
			t2d(i,j) = (V) (i*j+i);
		}

	t1d = t2d;
	for(int i=0;i<120; i++)
		for(int j=0;j<240; j++)
			BOOST_CHECK_EQUAL( (V) t1d(i,j) , (V) (i*j+i) );

	for(int i=0;i<120; i++)
		for(int j=0;j<240; j++){
			t1d(i,j) = (V) (i*j+i);
			t2d(i,j) = (V) 0;
		}

	t2d = t1d;
	for(int i=0;i<120; i++)
		for(int j=0;j<240; j++)
			BOOST_CHECK_EQUAL( (V) t2d(i,j) , (V) (i*j+i) );

	tensor<V,M,row_major>       t1d_from2d(t2d);
	for(int i=0;i<120; i++)
		for(int j=0;j<240; j++)
			BOOST_CHECK_EQUAL( (V) t1d_from2d(i,j) , (V) (i*j+i) );
}
BOOST_AUTO_TEST_CASE( mem2d )
{
	test_mem2d<float,host_memory_space>();
	test_mem2d<float,dev_memory_space>();

	test_mem2d<unsigned char,host_memory_space>();
	test_mem2d<unsigned char,dev_memory_space>();
}
BOOST_AUTO_TEST_CASE( tensor_view )
{
	linear_memory<float,host_memory_space,float*> l(100);
	const linear_memory<float,host_memory_space,float*> k(100, l.ptr(),true);
	BOOST_CHECK_EQUAL(l.ptr(), k.ptr());
	
	tensor<float,host_memory_space,column_major> a(extents[2][3][4]);
	tensor<float,host_memory_space,column_major> b(indices[2][3][4],a.ptr());
	BOOST_CHECK_EQUAL(a.ptr(), b.ptr());
}

BOOST_AUTO_TEST_SUITE_END()
