#define BOOST_TEST_MODULE example
#include <iostream>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <cuv_general.hpp>
#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>
#include <host_dia_matrix.hpp>
#include <vector_ops.hpp>
#include <convert.hpp>

using namespace cuv;
using namespace std;

struct Fix{
	Fix(){
		//initCUDA(1);
	}
	~Fix(){
		//exitCUDA();
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( convert_pushpull )
{
	dev_dense_matrix<float,column_major> dfc(32,16);
	host_dense_matrix<float,row_major>  hfr(16,32);
	dev_dense_matrix<float,row_major> dfr(32,16);
	host_dense_matrix<float,column_major>  hfc(16,32);

	// dfc <--> hfr
	convert(dfc, hfr);
	convert(hfr, dfc);

	// dfr <--> hfc
	convert(dfr, hfc);
	convert(hfc, dfr);
}

BOOST_AUTO_TEST_CASE( create_dev_plain2 )
{
	dev_dense_matrix<float,column_major> dfc(16,16); // "wrong" size
	host_dense_matrix<float,row_major>  hfr(16,32);
	convert(dfc, hfr);                               // should make dfc correct size
	convert(hfr, dfc);
	BOOST_CHECK( hfr.w() == dfc.h());
	BOOST_CHECK( hfr.h() == dfc.w());
}

BOOST_AUTO_TEST_CASE( create_dev_plain3 )
{
	dev_dense_matrix<float,column_major> dfc(32,16); 
	host_dense_matrix<float,row_major>  hfr(16,16);  // "wrong" size
	convert(hfr, dfc);
	convert(dfc, hfr);                               // should make dfc correct size
	BOOST_CHECK( hfr.w() == dfc.h());
	BOOST_CHECK( hfr.h() == dfc.w());
}

BOOST_AUTO_TEST_CASE( dia2host )
{
	host_dia_matrix<float>                 hdia(32,32,3,32);
	host_dense_matrix<float,column_major>  hdns(32,32);
	std::vector<int> off;
	off.push_back(0);
	off.push_back(1);
	off.push_back(-1);
	sequence(hdia.vec());
	hdia.set_offsets(off);
	//hdia.transpose(); // works, too
	convert(hdns,hdia);
	for(int i=0;i<hdns.h();i++){
		for(int j=0; j<hdns.w();j++){
			cout << hdns(i,j) << " ";
			BOOST_CHECK_CLOSE(hdns(i,j),hdia(i,j),0.01);
		}
		cout <<endl;
	}
}



BOOST_AUTO_TEST_SUITE_END()
