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
#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <boost/test/included/unit_test.hpp>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/tensor.hpp>
#include <cuv/basics/io.hpp>
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

template<class T>
void binary_save()
{
	T m(extents[2][3][4]);
	T n;

	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 3; ++j) {
			for (int k = 0; k < 4; ++k) {
				m(i,j,k)=i*j+k;
			}
		}
	}
	std::ofstream os("test.dat");
	boost::archive::binary_oarchive oa(os);
	oa << m;
	os.close();

	std::ifstream is("test.dat");
	boost::archive::binary_iarchive ia(is);
	ia >> n;

	BOOST_REQUIRE(n.ptr());
	BOOST_REQUIRE_EQUAL(m.ndim(), n.ndim());
	BOOST_REQUIRE(cuv::equal_shape(m,n));
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 3; ++j) {
			for (int k = 0; k < 4; ++k) {
				BOOST_CHECK_EQUAL(m(i,j,k), n(i,j,k));
			}
		}
	}
}

template<class T>
void empty_save()
{
	T m;
	T n;

	std::ofstream os("test.dat");
	boost::archive::binary_oarchive oa(os);
	oa << m;
	os.close();

	std::ifstream is("test.dat");
	boost::archive::binary_iarchive ia(is);
	ia >> n;

	BOOST_REQUIRE(!n.ptr());
	BOOST_REQUIRE_EQUAL(m.ndim(), n.ndim());
	BOOST_REQUIRE(cuv::equal_shape(m,n));
}


BOOST_AUTO_TEST_CASE( binary_save_test ) {
	binary_save<tensor<float,host_memory_space> >();
	binary_save<tensor<float,dev_memory_space> >();
}
BOOST_AUTO_TEST_CASE( empty_save_test ) {
	empty_save<tensor<float,host_memory_space> >();
	empty_save<tensor<float,dev_memory_space> >();
}

BOOST_AUTO_TEST_SUITE_END()
