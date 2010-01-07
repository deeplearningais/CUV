#define BOOST_TEST_MODULE example
#include <iostream>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <cuv_general.hpp>
#include <vector_ops.hpp>
#include <host_dia_matrix.hpp>
#include <convert.hpp>

using namespace std;
using namespace cuv;

static const int n=32;
static const int m=32;
static const int d=3;

struct Fix{
	host_dia_matrix<float> w;
	Fix()
	:  w(n,m,d,n) 
	{
		std::vector<int> off;
		off.push_back(0);
		off.push_back(1);
		off.push_back(-1);
		w.set_offsets(off);
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( trans )
{
	host_dia_matrix<float> lw (32,32,3,32);
	host_dia_matrix<float> lwt(32,32,3,32);
	std::vector<int> off;
	off.push_back(0);
	off.push_back(1);
	off.push_back(-1);
	lw.set_offsets(off);
	lwt.set_offsets(off);
	sequence(*lw.vec());
	sequence(*lwt.vec());
	BOOST_CHECK_CLOSE( lw(0,0), 0.f,  0.01 );
	BOOST_CHECK_CLOSE( lw(1,1), 1.f,  0.01 );
	BOOST_CHECK_CLOSE( lw(2,2), 2.f,  0.01 );
	BOOST_CHECK_CLOSE( lw(0,1), 32.f, 0.01 );
	BOOST_CHECK_CLOSE( lw(1,0), 65.f, 0.01 );
	BOOST_CHECK_CLOSE( lw(0,2), 0.f,  0.01 );
	BOOST_CHECK_CLOSE( lw(30,31), 62.f,  0.01 );
	lwt.transpose();
	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			BOOST_CHECK_CLOSE( lw(i,j), lwt(j,i),0.01 );
			cout << lw(i,j) <<" ";
		}
		cout<<endl;
	}
}



BOOST_AUTO_TEST_SUITE_END()
