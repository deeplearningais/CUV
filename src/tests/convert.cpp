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
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/tensor.hpp>
#include <cuv/basics/dia_matrix.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/convert/convert.hpp>

using namespace cuv;
using namespace std;

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
	Fix(){
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( convert_pushpull )
{
	tensor<float,dev_memory_space,column_major> dfc(extents[32][16]);
	tensor<float,host_memory_space,row_major>  hfr(extents[16][32]);
	tensor<float,dev_memory_space,row_major> dfr(extents[32][16]);
	tensor<float,host_memory_space,column_major>  hfc(extents[16][32]);

	// dfc <--> hfr
	convert(dfc, hfc);
	convert(hfc, dfc);

	// dfr <--> hfc
	convert(dfr, hfr);
	convert(hfr, dfr);
}

/*
 *BOOST_AUTO_TEST_CASE( create_dev_plain2 )
 *{
 *    tensor<float,dev_memory_space,column_major> dfc(extents[16][16]); // "wrong" size
 *    tensor<float,dev_memory_space,row_major>  hfr(extents[16][32]);
 *    convert(dfc, hfr);                               // should make dfc correct size
 *    convert(hfr, dfc);
 *    BOOST_CHECK( hfr.shape()[1] == dfc.shape()[0]);
 *    BOOST_CHECK( hfr.shape()[0] == dfc.shape()[1]);
 *}
 */
/*
 *
 *BOOST_AUTO_TEST_CASE( create_dev_plain3 )
 *{
 *    tensor<float,dev_memory_space,column_major> dfc(extents[32][16]); 
 *    tensor<float,host_memory_space,row_major>  hfr(extents[16][16]);  // "wrong" size
 *    convert(hfr, dfc);
 *    convert(dfc, hfr);                               // should make dfc correct size
 *    BOOST_CHECK( hfr.shape()[1] == dfc.shape()[0]);
 *    BOOST_CHECK( hfr.shape()[0] == dfc.shape()[1]);
 *}
 */

BOOST_AUTO_TEST_CASE( dia2host )
{
	dia_matrix<float,host_memory_space>                 hdia(32,32,3,32);
	tensor<float,host_memory_space,column_major>  hdns(extents[32][32]);
	std::vector<int> off;
	off.push_back(0);
	off.push_back(1);
	off.push_back(-1);
	sequence(hdia.vec());
	hdia.set_offsets(off);
	//hdia.transpose(); // works, too
	convert(hdns,hdia);
	for(int i=0;i<hdns.shape()[0];i++){
		for(int j=0; j<hdns.shape()[1];j++){
			cout << hdns(i,j) << " ";
			BOOST_CHECK_CLOSE((float)hdns(i,j),(float)hdia(i,j),0.01);
		}
		cout <<endl;
	}
}



BOOST_AUTO_TEST_SUITE_END()
