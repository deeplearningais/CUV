#define BOOST_TEST_MODULE example
#include <iostream>
#include <fstream>

#include <cuv_test.hpp>
#include <cuv_general.hpp>
#include <vector_ops.hpp>
#include <host_dia_matrix.hpp>
#include <dev_dia_matrix.hpp>
#include <convert.hpp>

using namespace std;
using namespace cuv;

static const int n=32;
static const int m=16;
static const int d=3;
static const int rf=1;


struct Fix{
	host_dia_matrix<float> w;
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


BOOST_AUTO_TEST_CASE( spmv_dia2dense )
{
	// hostdia->hostdense
	host_dense_matrix<float> w2(n,m);
	fill(w2.vec(),-1);
	convert(w2,w);
	MAT_CMP(w,w2,0.1);
	cout << w <<w2;
}

BOOST_AUTO_TEST_CASE( spmv_host2dev )
{
	// host->dev
	dev_dia_matrix<float> w2(n,m,w.num_dia(),w.stride(),rf);
	convert(w2,w);
	MAT_CMP(w,w2,0.1);
	fill(w.vec(),0);

	// dev->host
	convert(w,w2);
	MAT_CMP(w,w2,0.1);
}



BOOST_AUTO_TEST_SUITE_END()
