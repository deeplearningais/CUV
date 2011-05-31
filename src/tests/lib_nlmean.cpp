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

#define BOOST_TEST_MODULE lib_intimg
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <cuv/tools/cuv_general.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/libs/nlmeans/nlmeans.hpp>
#include <cuv/tools/timing.hpp>
#include <cuv/tools/cuv_test.hpp>

using namespace cuv;

#define MEASURE_TIME(MSG, OPERATION, ITERS)     \
	float MSG;                                  \
	if(1){                                      \
		Timing tim;                             \
		for(int i=0;i<ITERS;i++){               \
			OPERATION ;                         \
		}                                       \
			safeThreadSync();               \
		tim.update(ITERS);                      \
		printf("%s [%s] took %4.4f us/pass\n", #MSG, #OPERATION, 1000000.0f*tim.perf()); \
		MSG = 1000000.0f*tim.perf();            \
	}

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
 * @brief create dense device matrix.
 */
BOOST_AUTO_TEST_CASE( test_nlmeans )
{
	tensor<float,host_memory_space,row_major> mh(extents[256][256][256]);
	FILE* pFile = fopen("/home/VI/staff/schulz/checkout/git/wurzel/data/L2_17aug-upsampled.dat","r");
	unsigned int s = fread(mh.ptr(),sizeof(float),mh.size(),pFile);
	fclose(pFile);
	//cuvAssert(s==mh.memsize());
	typedef tensor<float,dev_memory_space,row_major, memory2d_tag> dev_t;
	//typedef tensor<float,dev_memory_space,row_major> dev_t;
	dev_t md, md2;
	md  = mh;
	md2 = md;
	libs::nlmeans::filter_nlmean(md2,md);
	mh = md2;
	pFile = fopen("nlmeanresult.dat","w");
	fwrite(mh.ptr(), sizeof(float), mh.size(), pFile);
	
	
}

BOOST_AUTO_TEST_SUITE_END()

