
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


// TODO: Refactor this: its also in export_matrix_ops.cpp
template<class MS, class V,class M, class I>
struct ms_type {
};
template<class V,class M, class I>
struct ms_type<dev_memory_space,V,M,I> {
	typedef dev_dense_matrix<V,M,I> type;
};
template<class V,class M, class I>
struct ms_type<host_memory_space,V,M,I> {
	typedef host_dense_matrix<V,M,I> type;
};
template<class Mat, class NewVT>
struct switch_value_type{
	typedef typename ms_type<typename matrix_traits<Mat>::memory_space_type,NewVT, typename Mat::memory_layout, typename Mat::index_type>::type type;
};
// end: to be refactored


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
void export_super_to_max(){
	typedef typename switch_value_type<M,int>::type Mint;
	def("super_to_max",(void (*)(M&,M&, int, int, Mint*,M*))super_to_max<typename M::value_type, typename M::memory_layout, typename M::index_type>, (
															arg("dst"),
															arg("img"),
															arg("poolsize"),
															arg("overlap"),
															arg("indices")=object(),
															arg("filter")=object())
														);


}

template <class M>
void export_padding_ops(){
	def("copy_into",(void (*)(M&,M&, int))copy_into<typename M::value_type, typename M::memory_layout, typename M::index_type>, (
															arg("dst"),
															arg("img"),
															arg("padding"))
														);

	def("strip_padding",(void (*)(M&,M&, unsigned int))strip_padding<typename M::value_type, typename M::memory_layout, typename M::index_type>, (
															arg("dst"),
															arg("img"),
															arg("padding"))
														);
}

template <class M, class V>
void export_row_ncopy(){
	def("row_ncopy",(void (*)(M&,V&, unsigned int))row_ncopy<typename M::value_type, typename M::memory_layout, typename M::index_type>, (
															arg("dst"),
															arg("img"),
															arg("rows")));
	def("filter_inverse",(void (*)(M&,M&, unsigned int))filter_inverse<typename M::value_type, typename M::memory_layout, typename M::index_type>, (
																arg("dst"),
																arg("filter"),
																arg("fs")));


}

void export_convolution_ops(){
	export_convolve< host_dense_matrix<float,row_major> >();
	export_convolve< dev_dense_matrix<float,row_major> >();
	export_super_to_max< host_dense_matrix<float,row_major> >();
	export_super_to_max< dev_dense_matrix<float,row_major>  >();
	export_padding_ops< host_dense_matrix<float,row_major> >();
	export_padding_ops< host_dense_matrix<float,row_major>  >();
	export_padding_ops< dev_dense_matrix<float,row_major>  >();
	export_row_ncopy< dev_dense_matrix<float,row_major>, dev_vector<float>  >();
}


