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

template<class V, class M, class L>
void export_kernels(){
        typedef tensor<V,M,L> R;
        // pairwise euclidean distance between two datasets
        def("pdist2",(void(*)(R&, const R&, const R&, const bool &)) libs::kernels::pairwise_distance_l2<V,M,L>,(arg("dist"),arg("X"),arg("Y"),arg("squared")=false));
        def("pdist2",(R(*)(const R&, const R&, const bool &)) libs::kernels::pairwise_distance_l2<V,M,L>,(arg("X"),arg("Y"),arg("squared")=false));
}

void export_libs_kernels(){
	//export_kernels<float,column_major,host_memory_space,unsigned int>();
	export_kernels<float,dev_memory_space,row_major>();
	export_kernels<float,dev_memory_space,column_major>();
}
