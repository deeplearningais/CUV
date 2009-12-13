#define BOOST_TEST_MODULE example
#include <boost/test/included/unit_test.hpp>

#include <cuv_general.hpp>
#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>
#include <convert.hpp>

using namespace cuv;

struct Fix{
	Fix(){
		//initCUDA(1);
	}
	~Fix(){
		//exitCUDA();
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( create_dev_plain )
{
	dev_dense_matrix<float,column_major> dfc(32,16);
	host_dense_matrix<float,row_major>  hfr(16,32);
	dev_dense_matrix<float,row_major> dfr(32,16);
	host_dense_matrix<float,column_major>  hfc(16,32);
	convert(dfc, hfr);
	convert(hfr, dfc);
}



BOOST_AUTO_TEST_SUITE_END()
