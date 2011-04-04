#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <pyublas/numpy.hpp>

#include <cuv/basics/dense_matrix.hpp>
#include <cuv/libs/kernels/kernels.hpp>

//using namespace std;
using namespace boost::python;
using namespace cuv;
using namespace cuv::libs::kernels;
namespace ublas = boost::numeric::ublas;

template<class V, class L, class M, class I>
void export_kernels(){
	typedef dense_matrix<V,L,M,I> mat;
	def("pairwise_distance",pairwise_distance<mat>,(arg("distances"),arg("X"),arg("Y")));
}

void export_libs_kernels(){
	//export_kernels<float,column_major,host_memory_space,unsigned int>();
	export_kernels<float,dev_memory_space,row_major,unsigned int>();
}
