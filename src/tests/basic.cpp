/** 
 * @file basic.cpp
 * @brief tests construction and access of basic vectors and matrices
 * @author Hannes Schulz
 * @date 2010-03-21
 */
#define BOOST_TEST_MODULE example
#include <boost/test/included/unit_test.hpp>

#include <cuv_general.hpp>
#include <dense_matrix.hpp>
#include <convert.hpp>

using namespace cuv;

struct Fix{
	static const int N=256;
	vector<float,dev_memory_space> v;
	vector<float,host_memory_space> w;
	Fix()
	:   v(N)
	,   w(N)
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
BOOST_AUTO_TEST_CASE( create_dev_plain )
{
	dense_matrix<float,column_major,dev_memory_space> m(16,16);
}

/** 
 * @test 
 * @brief view on a device vector (at same position in memory).
 */
BOOST_AUTO_TEST_CASE( create_dev_view )
{
	dense_matrix<float,column_major,dev_memory_space> m(16,16);
	dense_matrix<float,column_major,dev_memory_space> m2(16,16,new vector<float,dev_memory_space>(m.n(), m.ptr(), true));
}

/** 
 * @test 
 * @brief create a matrix using same width/height as another.
 */
BOOST_AUTO_TEST_CASE( create_dev_from_mat )
{
	dense_matrix<float,column_major,dev_memory_space> m(16,16);
	dense_matrix<float,column_major,dev_memory_space> m2(&m);
}

/** 
 * @test 
 * @brief creation of dense host matrix and a view.
 */
BOOST_AUTO_TEST_CASE( create_host )
{
	dense_matrix<float,column_major,host_memory_space> m(16,16);
	dense_matrix<float,column_major,host_memory_space> m2(16,16,new vector<float,host_memory_space>(m.n(),m.ptr(),true));
}

/** 
 * @test 
 * @brief setting and getting for device and host vectors.
 */
BOOST_AUTO_TEST_CASE( set_vector_elements )
{
	for(int i=0; i < N; i++) {
		v.set(i, (float) i/N);
		w.set(i, (float) i/N);
	}
	//convert(w,v);
	for(int i=0; i < N; i++) {
		BOOST_CHECK_EQUAL(v[i], (float) i/N );
		BOOST_CHECK_EQUAL(w[i], (float) i/N );
	}
}


BOOST_AUTO_TEST_SUITE_END()
