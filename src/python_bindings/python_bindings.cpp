#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>


#include <cuv_general.hpp>
#include <random.hpp>

using namespace boost::python;
using namespace cuv;

void export_vector();
//void export_matrix();

BOOST_PYTHON_MODULE(cuv_python){
	def("initCUDA", initCUDA);
	def("exitCUDA", exitCUDA);
	def("initialize_mersenne_twister_seeds", initialize_mersenne_twister_seeds);
	export_vector();
	//export_matrix();
}
