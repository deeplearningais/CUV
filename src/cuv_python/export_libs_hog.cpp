#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <pyublas/numpy.hpp>

#include <cuv/basics/tensor.hpp>
#include <cuv/libs/hog/hog.hpp>

//using namespace std;
using namespace boost::python;
using namespace cuv;
using namespace cuv::libs::hog;
namespace ublas = boost::numeric::ublas;

template<class V, class M>
void export_hog(){
	def("hog",hog<V,M>, (arg("descriptors"),arg("src"),arg("spatial_pooling")));
}


void export_libs_hog(){
	export_hog<float,host_memory_space>();
	export_hog<float,dev_memory_space>();
}
