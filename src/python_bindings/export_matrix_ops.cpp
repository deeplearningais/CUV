//*LB*
// Copyright (c) 2010, University of Bonn, Institute for Computer Science VI
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of the University of Bonn 
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*LE*






#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <pyublas/numpy.hpp>
#include  <boost/type_traits/is_base_of.hpp>

#include <dense_matrix.hpp>
#include <matrix_ops/matrix_ops.hpp>
#include <matrix_ops/rprop.hpp>
#include <convert.hpp>
#include <convolution_ops/convolution_ops.hpp>

//using namespace std;
using namespace boost::python;
using namespace cuv;
namespace ublas = boost::numeric::ublas;

//template<class MS, class V,class M, class I>
//struct ms_type {
//};
//template<class V,class M, class I>
//struct ms_type<dev_memory_space,V,M,I> {
	//typedef dev_dense_matrix<V,M,I> type;
//};
//template<class V,class M, class I>
//struct ms_type<host_memory_space,V,M,I> {
	//typedef host_dense_matrix<V,M,I> type;
//};

template<class Mat, class NewVT>
struct switch_value_type{
	typedef dense_matrix<NewVT, typename Mat::memory_layout, typename Mat::memory_space_type, typename Mat::index_type> type;
};

template<class VT, class MST, class IT>
boost::python::tuple matrix_arg_max(dense_matrix<VT, row_major, MST, IT>& mat){
	IT idx = arg_max(mat.vec());	
	return boost::python::make_tuple(idx / mat.w(), idx % mat.w());
}
template<class VT, class MST, class IT>
boost::python::tuple matrix_arg_max(dense_matrix<VT, column_major, MST, IT>& mat){
	IT idx = arg_max(mat.vec());	
	return boost::python::make_tuple(idx % mat.h(), idx / mat.h());
}
template<class VT, class MST, class IT>
boost::python::tuple matrix_arg_min(dense_matrix<VT, row_major, MST, IT>& mat){
	IT idx = arg_min(mat.vec());	
	return boost::python::make_tuple(idx / mat.w(), idx % mat.w());
}
template<class VT, class MST, class IT>
boost::python::tuple matrix_arg_min(dense_matrix<VT, column_major, MST, IT>& mat){
	IT idx = arg_min(mat.vec());	
	return boost::python::make_tuple(idx % mat.h(), idx / mat.h());
}

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
		 typename M::memory_space_type,
		 typename M::index_type>);
	def("apply_nullary_functor",
	   (void (*)(M&,const NullaryFunctor&, const typename M::value_type&)) 
	   apply_0ary_functor<
	     typename M::value_type,
		 typename M::memory_layout,
		 typename M::memory_space_type,
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
		 typename M::memory_space_type,
		 typename M::index_type>);
	def("apply_scalar_functor",
	   (void (*)(M&,const ScalarFunctor&, const typename M::value_type&)) 
	   apply_scalar_functor<
	     typename M::value_type,
		 typename M::memory_layout,
		 typename M::memory_space_type,
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
		 typename M::memory_space_type,
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
		 typename M::memory_space_type,
		 typename M::index_type,
		 typename N::value_type>);
	def("apply_binary_functor",
	   (void (*)(M&, N&, const BinaryFunctor&, const typename M::value_type&)) 
	   apply_binary_functor<
	     typename M::value_type,
		 typename M::memory_layout,
		 typename M::memory_space_type,
		 typename M::index_type,
	     typename M::value_type,
		 typename N::value_type>);
	def("apply_binary_functor",
	   (void (*)(M&, N&, const BinaryFunctor&, const typename M::value_type&, const typename M::value_type&)) 
	   apply_binary_functor<
	     typename M::value_type,
		 typename M::memory_layout,
		 typename M::memory_space_type,
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
	def("max_pool",(void (*)(M&,M&,unsigned int,unsigned int,Mint*, M*))max_pooling<typename M::value_type,typename M::memory_layout,typename M::memory_space_type,typename M::index_type>, (arg("dst"),arg("img"),arg("poolSize"),arg("overlap"),arg("optional_indices_of_maxima")=object(),arg("filter")=object()));
	def("supersample",(void (*)(M&,M&,int,Mint*))supersample<typename M::value_type,typename M::memory_layout,typename M::memory_space_type,typename M::index_type>,(arg("dst"),arg("img"),arg("factor"),arg("optional_indices")=object()));
}

template <class M>
void export_reductions(){
	def("has_inf",(bool (*)(typename M::vec_type&)) has_inf<typename M::vec_type>);
	def("has_inf",(bool (*)(M&)) has_inf<typename M::value_type,typename M::memory_layout,typename M::memory_space_type,typename M::index_type>);
	def("has_nan",(bool (*)(typename M::vec_type&)) has_nan<typename M::vec_type>);
	def("has_nan",(bool (*)(M&)) has_nan<typename M::value_type,typename M::memory_layout,typename M::memory_space_type,typename M::index_type>);
	def("norm1",(float (*)(typename M::vec_type&)) norm1<typename M::vec_type>);
	def("norm1",(float (*)(M&)) norm1<typename M::value_type,typename M::memory_layout,typename M::memory_space_type,typename M::index_type>);
	def("norm2",(float (*)(typename M::vec_type&)) norm2<typename M::vec_type>);
	def("norm2",(float (*)(M&)) norm2<typename M::value_type,typename M::memory_layout,typename M::memory_space_type,typename M::index_type>);
	def("maximum",(float (*)(typename M::vec_type&)) maximum<typename M::vec_type>);
	def("maximum",(float (*)(M&)) maximum<typename M::value_type,typename M::memory_layout,typename M::memory_space_type,typename M::index_type>);
	def("minimum",(float (*)(typename M::vec_type&)) minimum<typename M::vec_type>);
	def("minimum",(float (*)(M&)) minimum<typename M::value_type,typename M::memory_layout,typename M::memory_space_type,typename M::index_type>);
	def("mean", (float (*)(M&)) mean<typename M::value_type, typename M::memory_layout, typename M::memory_space_type, typename M::index_type>);
	def("arg_max",  (tuple(*)(M&)) matrix_arg_max<typename M::value_type, typename M::memory_space_type, typename M::index_type>);
	def("arg_min",  (tuple(*)(M&)) matrix_arg_min<typename M::value_type, typename M::memory_space_type, typename M::index_type>);
	def("reduce_to_col", reduce_to_col<M,typename M::vec_type>,(arg("vector"),arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=1.f,arg("factor_old")=0.f));
	def("reduce_to_row", reduce_to_row<M,typename M::vec_type>,(arg("vector"),arg("matrix"),arg("factor_new")=1.f,arg("factor_old")=0.f));
}

template <class M>
void export_blas2(){
	def("matrix_plus_col", matrix_plus_col<M,typename M::vec_type>);
	def("matrix_times_col", matrix_times_col<M,typename M::vec_type>);
	def("matrix_divide_col", matrix_divide_col<M,typename M::vec_type>);
}

template <class M>
void export_blockview(){
	typedef typename M::index_type I;
	def("blockview",(M*(*)(M&,I,I,I,I))
			blockview<typename M::value_type, typename M::memory_layout,typename M::memory_space_type,typename M::index_type>,return_value_policy<manage_new_object>());
}

template <class M>
void export_learn_step(){
	def("learn_step_weight_decay",(void (*)(M&, M&, const float&, const float&)) learn_step_weight_decay<typename M::value_type,typename M::memory_layout,typename M::memory_space_type,typename M::index_type>);
	def("rprop",
			(void (*)(M&, M&, M&,M&, const float&))
			rprop<typename M::value_type, typename M::value_type, typename M::memory_layout,typename M::memory_space_type,typename M::index_type>,
			(arg ("W"), arg ("dW"), arg ("dW_old"), arg ("learnrate") ,arg("cost")=0));
	typedef typename M::vec_type V;
	def("learn_step_weight_decay",(void (*)(V&, V&, const float&, const float&)) learn_step_weight_decay<V>);
	def("rprop",(void (*)(V&, V&, V&,V&, const float&)) rprop<V,V>);
}

template<class T>
void
export_transpose(){
	def("transpose", (void (*)(T&,T&))transpose);
}

template<class M>
void
export_multinomial_sampling(){
	def("sample_multinomial",(void (*)(M&))sample_multinomial<typename M::value_type,typename M::memory_layout,typename M::memory_space_type, typename M::index_type>);
	def("first_pool",	(void (*)(M&, M&, typename M::index_type))first_pooling<typename M::value_type,typename M::memory_layout,typename M::memory_space_type, typename M::index_type>);
	def("first_pool_zeros",	(void (*)(M&,  typename M::index_type))first_pooling_zeros<typename M::value_type,typename M::memory_layout,typename M::memory_space_type, typename M::index_type>);
	def("grid_to_matrix",    (void (*)(M&,M&,int))grid_to_matrix<typename M::value_type,typename M::memory_layout,typename M::memory_space_type, typename M::index_type>);
	def("matrix_to_grid",    (void (*)(M&,M&,int))matrix_to_grid<typename M::value_type,typename M::memory_layout,typename M::memory_space_type, typename M::index_type>);
	def("prob_max_pooling",    (void (*)(typename M::vec_type&,M&,int,bool))prob_max_pooling<typename M::value_type,typename M::memory_layout,typename M::memory_space_type, typename M::index_type>, (arg("sums"),arg("detection_layer"),arg("poolSize"),arg("sample")));
	def("prob_max_pooling",    (void (*)(M&,int,bool))prob_max_pooling<typename M::value_type,typename M::memory_layout,typename M::memory_space_type, typename M::index_type>, (arg("detection_layer"),arg("poolSize"),arg("sample")));
}


void export_matrix_ops(){
    enum_<cuv::reduce_functor>("reduce_functor")
        .value("ADD", RF_ADD)
        .value("ADD_SQUARED", RF_ADD_SQUARED)
        .value("MIN", RF_MIN)
        .value("MAX", RF_MAX)
        ;
	typedef dense_matrix<float,column_major,dev_memory_space> fdev;
	typedef dense_matrix<float,column_major,host_memory_space> fhost;
	typedef dense_matrix<float,row_major,host_memory_space> fhostr;
	typedef dense_matrix<float,row_major,dev_memory_space> fdevr;
	typedef dense_matrix<unsigned char,column_major,dev_memory_space> udev;
	typedef dense_matrix<unsigned char,column_major,host_memory_space> uhost;
	typedef dense_matrix<int,column_major,dev_memory_space> idev;
	typedef dense_matrix<int,column_major,host_memory_space> ihost;

	export_blas3<fdev,fdev,fdev>();
	export_blas3<fhost,fhost,fhost>();
	export_blas3<fdevr,fdevr,fdevr>();
	export_blas3<fhostr,fhostr,fhostr>();
	export_nullary_functor<fhost>();
	export_nullary_functor<fdev>();
	export_nullary_functor<uhost>();
	export_nullary_functor<udev>();
	export_nullary_functor<ihost>();
	export_nullary_functor<idev>();
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
	export_learn_step<fhostr>();
	export_learn_step<fdevr>();
	export_blas2<fdev>();
	export_blas2<fhost>();
	export_blas2<fhostr>();
	export_blas2<fdevr>();
	export_blockview<fdev>();
	export_blockview<fhost>();
	export_blockview<fdevr>();
	//export_pooling<fhostr>();
	export_pooling<fdevr>();
	// transpose
	export_transpose<fhost>();
	export_transpose<fdev>();
	export_transpose<fhostr>();
	export_transpose<fdevr>();

	export_multinomial_sampling<dense_matrix<float,row_major,dev_memory_space> >();

}


