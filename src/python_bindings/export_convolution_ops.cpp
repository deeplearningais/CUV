
#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <pyublas/numpy.hpp>
#include  <boost/type_traits/is_base_of.hpp>
#include <dev_vector.hpp>
#include <host_vector.hpp>
#include <vector_ops/vector_ops.hpp>
#include <convert.hpp>
#include <convolution_ops/convolution_ops.hpp>


//using namespace std;
using namespace boost::python;
using namespace cuv;
namespace ublas = boost::numeric::ublas;


template <class M>
void export_convolve(){
	def("convolve",(void (*)(M&,M&, M&))convolve<typename M::value_type,typename M::memory_layout,typename M::index_type>, (
																							arg("dst"),
																							arg("img"),
																							arg("filter"))
																						);
	def("convolve2",(void (*)(M&,M&, M&, int))convolve2<typename M::value_type,typename M::memory_layout,typename M::index_type>, (
																							arg("dst"),
																							arg("img"),
																							arg("filter"),
																							arg("numFilters"))
																						);
	def("convolve3",(void (*)(M&,M&, M&))convolve3<typename M::value_type,typename M::memory_layout,typename M::index_type>, (
																							arg("dst"),
																							arg("img"),
																							arg("filter"))
																						);
}

template <class M>
void export_convolve_2(){
	def("convolve",(void (*)(M&,M&, M&))convolve<>, (
														arg("dst"),
														arg("img"),
														arg("filter"))
													);
	def("convolve2",(void (*)(M&,M&, M&, int))convolve2<>, (
																arg("dst"),
																arg("img"),
																arg("filter"),
																arg("numFilters"))
															);
	def("convolve3",(void (*)(M&,M&, M&))convolve3<>, (
														arg("dst"),
														arg("img"),
														arg("filter"))
													);
}

void export_convolution_ops(){
	export_convolve< host_dense_matrix<float,row_major> >();
	export_convolve< dev_dense_matrix<float,row_major> >();
}


