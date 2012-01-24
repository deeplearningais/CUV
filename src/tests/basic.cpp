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
	Fix()
	{
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )

BOOST_AUTO_TEST_CASE( create_lm )
{
    unsigned int N = 54;
    {
        linear_memory<float,host_memory_space> v(N);
        BOOST_CHECK_EQUAL(v.size(), N);
        BOOST_CHECK_NE(v.ptr(), (float*)NULL);
        v.dealloc();
        BOOST_CHECK_EQUAL(v.ptr(), (float*)NULL);
    }
    {
        linear_memory<float,dev_memory_space> v(N);
        BOOST_CHECK_EQUAL(v.size(), N);
        BOOST_CHECK_NE(v.ptr(), (float*)NULL);
        v.dealloc();
        BOOST_CHECK_EQUAL(v.ptr(), (float*)NULL);
    }

}

BOOST_AUTO_TEST_CASE( readwrite_lm )
{
    unsigned int N = 54;
    {
        linear_memory<float,host_memory_space> v(N);
        v[1] = 0;
        BOOST_CHECK_EQUAL(v[1], 0);
        v[1] = 1;
        BOOST_CHECK_EQUAL(v[1], 1);
    }
    {
        linear_memory<float,dev_memory_space> v(N);
        v[1] = 0;
        BOOST_CHECK_EQUAL(v[1], 0);
        v[1] = 1;
        BOOST_CHECK_EQUAL(v[1], 1);
    }

}

BOOST_AUTO_TEST_CASE( create_pm )
{
    unsigned int N = 54, M=97;
    {
        pitched_memory<float,host_memory_space> v(N,M);
        BOOST_CHECK_EQUAL(v.size(), N*M);
        BOOST_CHECK_EQUAL(v.rows(), N);
        BOOST_CHECK_EQUAL(v.cols(), M);
        BOOST_CHECK_GE(v.pitch(), M);
        BOOST_CHECK_NE(v.ptr(), (float*)NULL);
        v.dealloc();
        BOOST_CHECK_EQUAL(v.ptr(), (float*)NULL);
    }
    {
        pitched_memory<float,dev_memory_space> v(N,M);
        BOOST_CHECK_GE(v.size(), N*M);
        BOOST_CHECK_EQUAL(v.rows(), N);
        BOOST_CHECK_EQUAL(v.cols(), M);
        BOOST_CHECK_GE(v.pitch(), M);
        BOOST_CHECK_NE(v.ptr(), (float*)NULL);
        v.dealloc();
        BOOST_CHECK_EQUAL(v.ptr(), (float*)NULL);
    }

}

BOOST_AUTO_TEST_CASE( readwrite_pm )
{
    unsigned int N = 54, M=97;
    {
        pitched_memory<float,host_memory_space> v(N,M);
        v[1] = 0;
        BOOST_CHECK_EQUAL(v[1], 0);
        v[1] = 1;
        BOOST_CHECK_EQUAL(v[1], 1);
    }
    {
        pitched_memory<float,dev_memory_space> v(N,M);
        v[1] = 0;
        BOOST_CHECK_EQUAL(v[1], 0);
        v[1] = 1;
        BOOST_CHECK_EQUAL(v[1], 1);
    }

    {
        pitched_memory<float,host_memory_space> v(N,M);
        v(3,4) = 0;
        BOOST_CHECK_EQUAL(v(3,4), 0);
        v(3,4) = 1;
        BOOST_CHECK_EQUAL(v(3,4), 1);
    }
    {
        pitched_memory<float,dev_memory_space> v(N,M);
        v(3,4) = 0;
        BOOST_CHECK_EQUAL(v(3,4), 0);
        v(3,4) = 1;
        BOOST_CHECK_EQUAL(v(3,4), 1);
    }

}

/** 
 * @test
 * @brief create dense matrix.
 */
BOOST_AUTO_TEST_CASE( create_linear )
{
    unsigned int N=16,M=32;
	{
        tensor<float,dev_memory_space,row_major> m(extents[N][M]);
        BOOST_CHECK_EQUAL(m.size(),N*M);
        BOOST_CHECK_EQUAL(m.shape(0),N);
        BOOST_CHECK_EQUAL(m.shape(1),M);
        BOOST_CHECK_EQUAL(m.stride(0),M);
        BOOST_CHECK_EQUAL(m.stride(1),1);
	}

	{
        tensor<float,host_memory_space,row_major> m(extents[N][M]);
        BOOST_CHECK_EQUAL(m.size(),N*M);
        BOOST_CHECK_EQUAL(m.shape(0),N);
        BOOST_CHECK_EQUAL(m.shape(1),M);
        BOOST_CHECK_EQUAL(m.stride(0),M);
        BOOST_CHECK_EQUAL(m.stride(1),1);
	}

	{
        tensor<float,dev_memory_space,column_major> m(extents[N][M]);
        BOOST_CHECK_EQUAL(m.size(),N*M);
        BOOST_CHECK_EQUAL(m.shape(0),N);
        BOOST_CHECK_EQUAL(m.shape(1),M);
        BOOST_CHECK_EQUAL(m.stride(0),1);
        BOOST_CHECK_EQUAL(m.stride(1),N);
	}

	{
        tensor<float,host_memory_space,column_major> m(extents[N][M]);
        BOOST_CHECK_EQUAL(m.size(),N*M);
        BOOST_CHECK_EQUAL(m.shape(0),N);
        BOOST_CHECK_EQUAL(m.shape(1),M);
        BOOST_CHECK_EQUAL(m.stride(0),1);
        BOOST_CHECK_EQUAL(m.stride(1),N);
	}
}

/** 
 * @test
 * @brief create pitched matrix.
 */
BOOST_AUTO_TEST_CASE( create_pitched )
{
    unsigned int N=16,M=32;
	{
        tensor<float,dev_memory_space,row_major> m(extents[N][M],pitched_memory_tag());
        BOOST_CHECK_EQUAL(m.size(),N*M);
        BOOST_CHECK_EQUAL(m.shape(0),N);
        BOOST_CHECK_EQUAL(m.shape(1),M);
        BOOST_CHECK_GE(m.stride(0),M);
        BOOST_CHECK_EQUAL(m.stride(1),1);
	}

	{
        tensor<float,host_memory_space,row_major> m(extents[N][M],pitched_memory_tag());
        BOOST_CHECK_EQUAL(m.size(),N*M);
        BOOST_CHECK_EQUAL(m.shape(0),N);
        BOOST_CHECK_EQUAL(m.shape(1),M);
        BOOST_CHECK_GE(m.stride(0),M);
        BOOST_CHECK_EQUAL(m.stride(1),1);
	}

	{
        tensor<float,dev_memory_space,column_major> m(extents[N][M],pitched_memory_tag());
        BOOST_CHECK_EQUAL(m.size(),N*M);
        BOOST_CHECK_EQUAL(m.shape(0),N);
        BOOST_CHECK_EQUAL(m.shape(1),M);
        BOOST_CHECK_EQUAL(m.stride(0),1);
        BOOST_CHECK_GE(m.stride(1),N);
	}

	{
        tensor<float,host_memory_space,column_major> m(extents[N][M],pitched_memory_tag());
        BOOST_CHECK_EQUAL(m.size(),N*M);
        BOOST_CHECK_EQUAL(m.shape(0),N);
        BOOST_CHECK_EQUAL(m.shape(1),M);
        BOOST_CHECK_EQUAL(m.stride(0),1);
        BOOST_CHECK_GE(m.stride(1),N);
	}
}


/** 
 * @test 
 * @brief setting and getting for device and host vectors.
 */
BOOST_AUTO_TEST_CASE( set_vector_elements )
{
    static const unsigned int N = 145;
    static const unsigned int M = 97;
	tensor<float,host_memory_space> v(extents[N][M]);                     // linear memory
	tensor<float,dev_memory_space> w(extents[N][M],pitched_memory_tag()); // pitched memory
	for(int i=0; i < N; i++) {
		v[i]= (float)i/N;
		w[i]= (float)i/N;
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
