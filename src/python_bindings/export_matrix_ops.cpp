
#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <pyublas/numpy.hpp>
#include  <boost/type_traits/is_base_of.hpp>

#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>
#include <matrix_ops/matrix_ops.hpp>
#include <matrix_ops/rprop.hpp>
#include <convert.hpp>
#include <convolution_ops/convolution_ops.hpp>

//using namespace std;
using namespace boost::python;
using namespace cuv;
namespace ublas = boost::numeric::ublas;

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

template<class R, class S, class T>
void export_blas3() {
	def("prod",&prod<R,S,T>, (
				arg("C"), arg("A"), arg("B"), arg("transA")='n', arg("transB")='n', arg("factAB")=1.f, arg("factC")=0.f
				));
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
	typedef typename M::vec_type V;
	def("apply_nullary_functor",
	   (void (*)(V&,const NullaryFunctor&)) 
	   apply_0ary_functor<V>);
	def("apply_nullary_functor",
	   (void (*)(V&,const NullaryFunctor&, const typename V::value_type&)) 
	   apply_0ary_functor<V,typename V::value_type>);

	// convenience wrappers
	def("sequence", (void (*)(V&)) sequence);
	def("sequence", (void (*)(M&)) sequence);
	def("fill",     (void (*)(V&,const typename V::value_type&)) fill);
	def("fill",     (void (*)(M&,const typename V::value_type&)) fill);
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
	typedef typename M::vec_type V;
	def("apply_scalar_functor",
	   (void (*)(V&,const ScalarFunctor&)) 
	   apply_scalar_functor<V>);
	def("apply_scalar_functor",
	   (void (*)(V&,const ScalarFunctor&, const typename V::value_type&)) 
	   apply_scalar_functor<V,typename V::value_type>);
}

template<class M, class N>
void export_binary_functor_simple() {
	def("apply_binary_functor",
	   (void (*)(M&, N&, const BinaryFunctor&)) 
	   apply_binary_functor<
	     typename M::value_type,
		 typename M::memory_layout,
		 typename M::index_type,
		 typename N::value_type>);
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
	typedef typename M::vec_type V;
	typedef typename N::vec_type W;
	def("apply_binary_functor",
	   (void (*)(V&, W&, const BinaryFunctor&)) 
	   apply_binary_functor<V,W>);
	def("apply_binary_functor",
	   (void (*)(V&, W&, const BinaryFunctor&, const typename V::value_type&)) 
	   apply_binary_functor<V,W,typename V::value_type>);
	def("apply_binary_functor",
	   (void (*)(V&, W&, const BinaryFunctor&, const typename V::value_type&, const typename V::value_type&)) 
	   apply_binary_functor<
	     V,
		 W,
	     typename M::value_type>);
}

template <class M>
void export_pooling(){
	typedef typename switch_value_type<M,int>::type Mint;
	def("max_pool",(void (*)(M&,M&,unsigned int,unsigned int,Mint*))max_pooling<typename M::value_type,typename M::memory_layout,typename M::index_type>, (arg("dst"),arg("img"),arg("poolSize"),arg("overlap"),arg("optional_indices")=object()));
	def("supersample",(void (*)(M&,M&,int,Mint*))supersample<typename M::value_type,typename M::memory_layout,typename M::index_type>,(arg("dst"),arg("img"),arg("factor"),arg("optional_indices")=object()));
}

template <class M>
void export_reductions(){
	def("has_inf",(bool (*)(typename M::vec_type&)) has_inf<typename M::vec_type>);
	def("has_inf",(bool (*)(M&)) has_inf<typename M::value_type,typename M::memory_layout,typename M::index_type>);
	def("has_nan",(bool (*)(typename M::vec_type&)) has_nan<typename M::vec_type>);
	def("has_nan",(bool (*)(M&)) has_nan<typename M::value_type,typename M::memory_layout,typename M::index_type>);
	def("norm2",(float (*)(typename M::vec_type&)) norm2<typename M::vec_type>);
	def("norm2",(float (*)(M&)) norm2<typename M::value_type,typename M::memory_layout,typename M::index_type>);
	def("reduce_to_col", reduce_to_col<M,typename M::vec_type>,(arg("vector"),arg("matrix"),arg("factor_new")=1.f,arg("factor_old")=0.f));
}

template <class M>
void export_blas2(){
	def("matrix_plus_col", matrix_plus_col<M,typename M::vec_type>);
	def("matrix_times_col", matrix_times_col<M,typename M::vec_type>);
}

template <class M>
void export_blockview(){
	def("blockview",blockview<M>,return_value_policy<manage_new_object>());
}

template <class M>
void export_learn_step(){
	def("learn_step_weight_decay",(void (*)(M&, M&, const float&, const float&)) learn_step_weight_decay<typename M::value_type,typename M::memory_layout,typename M::index_type>);
	def("rprop",(void (*)(M&, M&, M&,M&)) rprop<typename M::value_type, typename M::value_type, typename M::memory_layout,typename M::index_type>);
	typedef typename M::vec_type V;
	def("learn_step_weight_decay",(void (*)(V&, V&, const float&, const float&)) learn_step_weight_decay<V>);
	def("rprop",(void (*)(V&, V&, V&,V&)) rprop<V,V>);
}

template<class T>
void
export_transpose(){
	def("transpose", (void (*)(T&,T&))transpose);
}

template<class M>
void
export_multinomial_sampling(){
	def("sample_multinomial",(void (*)(M&))sample_multinomial<typename M::value_type,typename M::memory_layout, typename M::index_type>);
	def("grid_to_matrix",    (void (*)(M&,M&,int))grid_to_matrix<typename M::value_type,typename M::memory_layout, typename M::index_type>);
	def("matrix_to_grid",    (void (*)(M&,M&,int))matrix_to_grid<typename M::value_type,typename M::memory_layout, typename M::index_type>);
	def("prob_max_pooling",    (void (*)(typename M::vec_type&,M&,int,bool))prob_max_pooling<typename M::value_type,typename M::memory_layout, typename M::index_type>, (arg("pooled_layer"),arg("detection_layer"),arg("poolSize"),arg("sample")));
	def("prob_max_pooling",    (void (*)(M&,int,bool))prob_max_pooling<typename M::value_type,typename M::memory_layout, typename M::index_type>, (arg("detection_layer"),arg("poolSize"),arg("sample")));
}


void export_matrix_ops(){
	typedef dev_dense_matrix<float,column_major> fdev;
	typedef host_dense_matrix<float,column_major> fhost;
	typedef host_dense_matrix<float,row_major> fhostr;
	typedef dev_dense_matrix<float,row_major> fdevr;
	typedef dev_dense_matrix<unsigned char,column_major> udev;
	typedef host_dense_matrix<unsigned char,column_major> uhost;

	export_blas3<fdev,fdev,fdev>();
	export_blas3<fhost,fhost,fhost>();
	export_nullary_functor<fhost>();
	export_nullary_functor<fdev>();
	export_scalar_functor<fhost>();
	export_scalar_functor<fdev>();
	export_binary_functor<fdev,fdev>();
	export_binary_functor<fhost,fhost>();
	export_nullary_functor<fhostr>();
	export_nullary_functor<fdevr>();
	export_scalar_functor<fhostr>();
	export_scalar_functor<fdevr>();
	export_binary_functor<fdevr,fdevr>();
	export_binary_functor<fhostr,fhostr>();
	export_binary_functor_simple<fhost,uhost>();
	export_binary_functor_simple<fdev,udev>();
	export_reductions<fhost>();
	export_reductions<fdev>();
	export_reductions<fhostr>();
	export_reductions<fdevr>();
	export_learn_step<fhost>();
	export_learn_step<fdev>();
	export_blas2<fdev>();
	export_blas2<fhost>();
	export_blas2<fhostr>();
	export_blas2<fdevr>();
	export_blockview<fdev>();
	export_blockview<fhost>();
	//export_pooling<fhostr>();
	export_pooling<fdevr>();
	// transpose
	export_transpose<fhost>();
	export_transpose<fdev>();
	export_transpose<fhostr>();
	export_transpose<fdevr>();

	export_multinomial_sampling<dev_dense_matrix<float,row_major> >();

}


