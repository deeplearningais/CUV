
#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <pyublas/numpy.hpp>
#include  <boost/type_traits/is_base_of.hpp>

#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>
#include <matrix_ops/matrix_ops.hpp>
#include <convert.hpp>

using namespace std;
using namespace boost::python;
using namespace cuv;
namespace ublas = boost::numeric::ublas;

template<class R, class S, class T>
void export_blas3() {
	def("prod",&prod<R,S,T>);
}

template<class M>
void export_nullary_functor() {
	def("apply_nullary_functor",
	   (void (*)(M&,const NullaryFunctor&)) 
	   apply_0ary_functor<
	     typename M::value_type,
		 typename M::memory_layout,
		 typename M::index_type>);
	def("apply_nullary_functor",
	   (void (*)(M&,const NullaryFunctor&, const typename M::value_type&)) 
	   apply_0ary_functor<
	     typename M::value_type,
		 typename M::memory_layout,
		 typename M::index_type,
		 typename M::value_type>);
}

template<class M>
void export_scalar_functor() {
	def("apply_scalar_functor",
	   (void (*)(M&,const ScalarFunctor&)) 
	   apply_scalar_functor<
	     typename M::value_type,
		 typename M::memory_layout,
		 typename M::index_type>);
	def("apply_scalar_functor",
	   (void (*)(M&,const ScalarFunctor&, const typename M::value_type&)) 
	   apply_scalar_functor<
	     typename M::value_type,
		 typename M::memory_layout,
		 typename M::index_type,
		 typename M::value_type>);
}

template<class M, class N>
void export_binary_functor() {
	def("apply_binary_functor",
	   (void (*)(M&, N&, const BinaryFunctor&)) 
	   apply_binary_functor<
	     typename M::value_type,
		 typename M::memory_layout,
		 typename M::index_type,
		 typename N::value_type>);
	def("apply_binary_functor",
	   (void (*)(M&, N&, const BinaryFunctor&, const typename M::value_type&)) 
	   apply_binary_functor<
	     typename M::value_type,
		 typename M::memory_layout,
		 typename M::index_type,
	     typename M::value_type,
		 typename N::value_type>);
	def("apply_binary_functor",
	   (void (*)(M&, N&, const BinaryFunctor&, const typename M::value_type&, const typename M::value_type&)) 
	   apply_binary_functor<
	     typename M::value_type,
		 typename M::memory_layout,
		 typename M::index_type,
	     typename M::value_type,
		 typename N::value_type>);
}

template <class M>
void export_reductions(){
	def("norm2",(float (*)(M&)) norm2<typename M::value_type,typename M::memory_layout,typename M::index_type>);
}

void export_matrix_ops(){
	typedef dev_dense_matrix<float,column_major> fdev;
	typedef host_dense_matrix<float,column_major> fhost;

	export_blas3<fdev,fdev,fdev>();
	export_blas3<fhost,fhost,fhost>();
	export_nullary_functor<fhost>();
	export_nullary_functor<fdev>();
	export_scalar_functor<fhost>();
	export_scalar_functor<fdev>();
	export_binary_functor<fdev,fdev>();
	export_binary_functor<fhost,fhost>();
	export_reductions<fhost>();
	export_reductions<fdev>();

}


