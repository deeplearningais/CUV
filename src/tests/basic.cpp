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





/** 
 * @file basic.cpp
 * @brief tests construction and access of basic vectors and matrices
 * @author Hannes Schulz
 * @date 2010-03-21
 */
#define BOOST_TEST_MODULE example
#include <boost/test/included/unit_test.hpp>

#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/tensor.hpp>
#include <cuv/basics/cuda_array.hpp>
#include <cuv/convert/convert.hpp>

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
	static const int N;
	tensor<float,dev_memory_space> v;
	tensor<float,host_memory_space> w;
	Fix()
	:   v(N)
	,   w(N)
	{
	}
	~Fix(){
	}
};
const int Fix::N = 256;


BOOST_FIXTURE_TEST_SUITE( s, Fix )


/** 
 * @test
 * @brief create dense device matrix.
 */
BOOST_AUTO_TEST_CASE( create_dev_plain )
{
	tensor<float,dev_memory_space,column_major> m(16,16);
}

/** 
 * @test 
 * @brief view on a device vector (at same position in memory).
 */
BOOST_AUTO_TEST_CASE( create_dev_view )
{
	tensor<float,dev_memory_space,column_major> m(16,16);
	//tensor<float,column_major,dev_memory_space> m2(16,16,new linear_memory<float,dev_memory_space>(m.n(), m.ptr(), true));
}

/** 
 * @test 
 * @brief create a matrix using same width/height as another.
 */
BOOST_AUTO_TEST_CASE( create_dev_from_mat )
{
	tensor<float,dev_memory_space,column_major> m(16,16);
	//tensor<float,column_major,dev_memory_space> m2(&m);
}

/** 
 * @test 
 * @brief creation of dense host matrix and a view.
 */
BOOST_AUTO_TEST_CASE( create_host )
{
	tensor<float,host_memory_space,column_major> m(16,16);
	//tensor<float,column_major,host_memory_space> m2(16,16,new vector<float,host_memory_space>(m.n(),m.ptr(),true));
}

/** 
 * @test 
 * @brief setting and getting for device and host vectors.
 */
BOOST_AUTO_TEST_CASE( set_vector_elements )
{
	for(int i=0; i < N; i++) {
		v[i]=(float) i/N;
		w[i]=(float) i/N;
	}
	//convert(w,v);
	for(int i=0; i < N; i++) {
		BOOST_CHECK_EQUAL(v[i], (float) i/N );
		BOOST_CHECK_EQUAL(w[i], (float) i/N );
	}
}

/** 
 * @test 
 * @brief allocating and destroying a cuda_array
 */
BOOST_AUTO_TEST_CASE( cuda_array_alloc )
{
	cuda_array<float,dev_memory_space> ca(1024,768);
	BOOST_CHECK(ca.ptr());
}

/** 
 * @test 
 * @brief copying a dense matrix to a cuda_array
 */
BOOST_AUTO_TEST_CASE( cuda_array_assign )
{
	cuda_array<float,dev_memory_space> ca(1024,768);
	tensor<float,dev_memory_space,row_major> dm(1024,768);
	ca.assign(dm);
}


BOOST_AUTO_TEST_SUITE_END()
