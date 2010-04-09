#define BOOST_TEST_MODULE example
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <cuv_general.hpp>
#include <dense_matrix.hpp>
#include <convert.hpp>
#include <matrix_ops.hpp>
#include <matrix_ops/rprop.hpp>
#include <cuv_test.hpp>
#include <../random/random.hpp>

using namespace cuv;

struct Fix{
	Fix() {
	}
	~Fix(){
	}
};

BOOST_FIXTURE_TEST_SUITE( s, Fix )

BOOST_AUTO_TEST_CASE( mem_dealloc )
{
	for(int i=0; i<100000000; i++) {
		dense_matrix<float, row_major, host_memory_space> c(1000, 100000);
		dense_matrix<float, row_major, dev_memory_space> d(1000, 100000);
	}
}


BOOST_AUTO_TEST_SUITE_END()
