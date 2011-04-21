#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <pyublas/numpy.hpp>

#include <cuv/basics/tensor.hpp>
#include <cuv/libs/kernels/kernels.hpp>

//using namespace std;
using namespace boost::python;
using namespace cuv;
using namespace cuv::libs::kernels;
namespace ublas = boost::numeric::ublas;

template<class V, class L, class M>
void export_kernels(){
	def("pairwise_distance_l2",pairwise_distance_l2<V,L,M>,(arg("distances"),arg("X"),arg("Y")));
}

void export_libs_kernels(){
	//export_kernels<float,column_major,host_memory_space,unsigned int>();
	export_kernels<float,dev_memory_space,row_major>();
	export_kernels<float,dev_memory_space,column_major>();
}
