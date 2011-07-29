#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <cuv/basics/io.hpp>
#include <cuv.hpp>
using namespace cuv;

int main(void){
    tensor<float,host_memory_space>  d(256);  // reserves space in device memory
    sequence(d);                        // fill device vector with a sequence

    {       // save matrix to file
	    std::ofstream os("/tmp/serialization_test.cuv");
	    boost::archive::binary_oarchive oa(os);
	    oa << d;
    }

    d = 0.f; // reset matrix to zero

    {       // read matrix from file
	    std::ifstream is("/tmp/serialization_test.cuv");
	    boost::archive::binary_iarchive ia(is);
	    ia >> d;
    }
    assert( (float)d[255] == 255.f );
}
