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
#include <numeric>
#include <boost/test/included/unit_test.hpp>
#include <cuv/tools/cuv_test.hpp>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/tensor.hpp>
#include <cuv/libs/kmeans/kmeans.hpp>
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


template<class M>
void
test_sort_by_idx_impl(){
	unsigned int data_num = 16;
	unsigned int data_len = 4;
	tensor<float,M> data(extents[data_num][data_len]);
	tensor<unsigned int, M> indices(data_num);

	std::vector<unsigned int> hidx(data_num);
	for (unsigned int i = 0; i < data_num; ++i)
		hidx[i] = i;
	std::random_shuffle(hidx.begin(), hidx.end());

	for (unsigned int i = 0; i < data_num; ++i)
		indices[i] = hidx[i];

	for (unsigned int i = 0; i < data_num; ++i)
	{
		tensor<float,M> tmp(cuv::indices[i][index_range(0,data_len)],data); 
		tmp = (float) hidx[i];
	}

	tensor<float, M> sorted(data.shape());
	cuv::libs::kmeans::sort_by_index(sorted,indices,data);


	for (unsigned int i = 0; i < data_num; ++i)
	{
		for (unsigned int j = 0; j < data_len; ++j)
		{
			BOOST_CHECK_EQUAL(sorted(i,j),i);
		}
		BOOST_CHECK_EQUAL(indices(i),i);
	}

}

BOOST_FIXTURE_TEST_SUITE( s, Fix )

/** 
 * @test
 * @brief test reordering of a dataset according to indices
 */
BOOST_AUTO_TEST_CASE( test_sort_by_idx_host )
{
	test_sort_by_idx_impl<host_memory_space>();
}

/** 
 * @test
 * @brief test reordering of a dataset according to indices
 */
BOOST_AUTO_TEST_CASE( test_sort_by_idx_dev )
{
	test_sort_by_idx_impl<dev_memory_space>();
}


BOOST_AUTO_TEST_SUITE_END()
