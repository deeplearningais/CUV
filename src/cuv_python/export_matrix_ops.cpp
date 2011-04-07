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

#include <cuv/basics/dense_matrix.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/tensor_ops/rprop.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>
#include <float.h>
//using namespace std;
using namespace boost::python;
using namespace cuv;
namespace ublas = boost::numeric::ublas;

template<class VT, class MST, class IT>
boost::python::tuple matrix_arg_max(dense_matrix<VT, MST, row_major, IT>& mat){
	IT idx = arg_max(mat.vec());	
	return boost::python::make_tuple(idx / mat.w(), idx % mat.w());
}
template<class VT, class MST, class IT>
boost::python::tuple matrix_arg_max(dense_matrix<VT, MST, column_major, IT>& mat){
	IT idx = arg_max(mat.vec());	
	return boost::python::make_tuple(idx % mat.h(), idx / mat.h());
}
template<class VT, class MST, class IT>
boost::python::tuple matrix_arg_min(dense_matrix<VT, MST, row_major, IT>& mat){
	IT idx = arg_min(mat.vec());	
	return boost::python::make_tuple(idx / mat.w(), idx % mat.w());
}
template<class VT, class MST, class IT>
boost::python::tuple matrix_arg_min(dense_matrix<VT, MST, column_major, IT>& mat){
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
	   (void (*)(typename M::tensor_type&,const NullaryFunctor&)) 
	   apply_0ary_functor< typename M::value_type, typename M::memory_space_type>);
	def("apply_nullary_functor",
	   (void (*)(typename M::tensor_type&,const NullaryFunctor&, const typename M::value_type&)) 
	   apply_0ary_functor< typename M::value_type, typename M::memory_space_type, typename M::value_type>);

	// convenience wrappers
	def("sequence", (void (*)(typename M::tensor_type&)) sequence);
	def("fill",     (void (*)(typename M::tensor_type&,const typename M::value_type&)) fill);
}

template<class M>
void export_scalar_functor() {
	// in place
	//def("apply_scalar_functor",
	   //(void (*)(typename M::tensor_type&,const ScalarFunctor&)) 
	   //apply_scalar_functor< typename M::value_type, typename M::memory_space_type >);
	//def("apply_scalar_functor",
	   //(void (*)(typename M::tensor_type&,const ScalarFunctor&, const typename M::value_type&)) 
	   //apply_scalar_functor< typename M::value_type, typename M::memory_space_type, typename M::value_type>);
        def("apply_scalar_functor",
           (void (*)(M&,const ScalarFunctor&)) 
           apply_scalar_functor<M>);
        def("apply_scalar_functor",
           (void (*)(M&,const ScalarFunctor&, const typename M::value_type&)) 
           apply_scalar_functor<M>);
        // not in place
        def("apply_scalar_functor",
           (void (*)(M&, const M&, const ScalarFunctor&)) 
           apply_scalar_functor<M>);
        def("apply_scalar_functor",
           (void (*)(M&,const M&,const ScalarFunctor&, const typename M::value_type&)) 
           apply_scalar_functor<M>);
}

template<class M, class N>
void export_binary_functor_simple() {
	def("apply_binary_functor",
	   (void (*)(M&, const N&, const BinaryFunctor&)) 
	   apply_binary_functor<
	     typename M::value_type,
		 typename M::memory_space_type,
		 typename N::value_type>);
}
template<class M, class N>
void export_binary_functor() {
	//def("apply_binary_functor",
	   //(void (*)(M&, const N&, const BinaryFunctor&)) 
	   //apply_binary_functor<
	     //typename M::value_type,
		 //typename M::memory_space_type,
		 //typename N::value_type>);
	//def("apply_binary_functor",
	   //(void (*)(M&, const N&, const BinaryFunctor&, const typename M::value_type&)) 
	   //apply_binary_functor<
	     //typename M::value_type,
		 //typename M::memory_space_type,
                 //typename M::value_type,
		 //typename N::value_type>);
	//def("apply_binary_functor",
	   //(void (*)(M&, const N&, const BinaryFunctor&, const typename M::value_type&, const typename M::value_type&)) 
	   //apply_binary_functor<
	     //typename M::value_type,
		 //typename M::memory_space_type,
	     //typename M::value_type,
		 //typename N::value_type>);
        def("apply_binary_functor",
           (void (*)(M&,const N&, const BinaryFunctor&)) 
           apply_binary_functor<M,N>);
        def("apply_binary_functor",
           (void (*)(M&, const N&, const BinaryFunctor&, const typename M::value_type&)) 
           apply_binary_functor<M,N>);
        def("apply_binary_functor",
           (void (*)(M&, const N&, const BinaryFunctor&, const typename M::value_type&, const typename M::value_type&)) 
           apply_binary_functor< M, N>);
}

//template <class M>
//void export_pooling(){
	//typedef typename switch_value_type<M,int>::type Mint;
	//def("max_pool",(void (*)(M&,M&,unsigned int,unsigned int,Mint*, M*))max_pooling<typename M::value_type,typename M::memory_layout,typename M::memory_space_type,typename M::index_type>, (arg("dst"),arg("img"),arg("poolSize"),arg("overlap"),arg("optional_indices_of_maxima")=object(),arg("filter")=object()));
	//def("supersample",(void (*)(M&,M&,int,Mint*))supersample<typename M::value_type,typename M::memory_layout,typename M::memory_space_type,typename M::index_type>,(arg("dst"),arg("img"),arg("factor"),arg("optional_indices")=object()));
//}

//template <class V, class MS>
//void export_argmax_vec(){
	//typedef dense_matrix<V,MS,column_major> Mc;
	//typedef dense_matrix<V,MS,row_major> Mr;
	//typedef tensor<int,MS> Vecint;
	//def("argmax_to_row",  (void (*)(Vecint&,const Mc&)) argmax_to_row<Vecint, Mc>);
	//def("argmax_to_col",  (void (*)(Vecint&,const Mr&)) argmax_to_column<Vecint, Mr>);
	//typedef tensor<float,MS> Vecf;
	//def("argmax_to_row",  (void (*)(Vecf&,const Mc&)) argmax_to_row<Vecf, Mc>);
	//def("argmax_to_col",  (void (*)(Vecf&,const Mr&)) argmax_to_column<Vecf, Mr>);
//}
template <class M>
void export_reductions(){
	//typedef typename switch_value_type<M, typename M::index_type>::type::vec_type idx_vec;
	//typedef typename switch_value_type<M, float>::type::vec_type float_vec;
	typedef typename M::value_type value_type;
	def("has_inf",(bool (*)(const typename M::tensor_type&)) has_inf<typename M::value_type,typename M::memory_space_type>);
	def("has_nan",(bool (*)(const typename M::tensor_type&)) has_nan<typename M::value_type,typename M::memory_space_type>);
	def("sum",(float (*)(const typename M::tensor_type&)) sum<typename M::value_type,typename M::memory_space_type>);
	def("norm1",(float (*)(const typename M::tensor_type&)) norm1<typename M::value_type,typename M::memory_space_type>);
	def("norm2",(float (*)(const typename M::tensor_type&)) norm2<typename M::value_type,typename M::memory_space_type>);
	def("maximum",(float (*)(const typename M::tensor_type&)) maximum<typename M::value_type,typename M::memory_space_type>);
	def("minimum",(float (*)(const typename M::tensor_type&)) minimum<typename M::value_type,typename M::memory_space_type>);
	def("mean", (float (*)(const typename M::tensor_type&)) mean<typename M::value_type,typename M::memory_space_type>);
	//def("arg_max",  (tuple(*)( M&)) matrix_arg_max<typename M::value_type, typename M::memory_space_type, typename M::index_type>);
	//def("arg_min",  (tuple(*)( M&)) matrix_arg_min<typename M::value_type, typename M::memory_space_type, typename M::index_type>);
	//def("reduce_to_col", reduce_to_col<M,typename M::vec_type>,(arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));
	//def("reduce_to_row", reduce_to_row<M,typename M::vec_type>,(arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));
	//def("reduce_to_col", reduce_to_col<M,idx_vec>,(arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));
	//def("reduce_to_row", reduce_to_row<M,idx_vec>,(arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));

	//def("reduce_to_col", reduce_to_col<M,float_vec>,(arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));
	//def("reduce_to_row", reduce_to_row<M,float_vec>,(arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));
}

template <class M, class V2>
void export_blas2(){
	//def("matrix_plus_col", matrix_plus_col<M,typename switch_value_type<M,V2>::type::vec_type>);
	//def("matrix_times_col", matrix_times_col<M,typename switch_value_type<M,V2>::type::vec_type>);
	//def("matrix_divide_col", matrix_divide_col<M,typename switch_value_type<M,V2>::type::vec_type>);
	//def("matrix_plus_row", matrix_plus_row<M,typename switch_value_type<M,V2>::type::vec_type>);
	//def("matrix_times_row", matrix_times_row<M,typename switch_value_type<M,V2>::type::vec_type>);
	//def("matrix_divide_row", matrix_divide_row<M,typename switch_value_type<M,V2>::type::vec_type>);
}

template <class M>
void export_bitflip(){
	typedef typename M::index_type I;
	//def("bitflip",(void(*)(M&,I))
			//bitflip<typename M::value_type, typename M::memory_layout,typename M::memory_space_type,typename M::index_type>);
}
template <class M>
void export_blockview(){
	typedef typename M::index_type I;
	def("blockview",(M*(*)(M&,I,I,I,I))
			blockview<typename M::value_type,typename M::memory_space_type, typename M::memory_layout,typename M::index_type>,return_value_policy<manage_new_object>());
}

template <class M>
void export_learn_step(){
	def("learn_step_weight_decay",(void (*)(typename M::tensor_type&, typename M::tensor_type&, const float&, const float&)) learn_step_weight_decay<typename M::tensor_type>);
	//def("rprop",
			//(void (*)(M&, M&, M&,M&, const float&))
			//rprop<typename M::value_type, typename M::value_type, typename M::memory_layout,typename M::memory_space_type,typename M::index_type>,
			//(arg ("W"), arg ("dW"), arg ("dW_old"), arg ("learnrate") ,arg("cost")=0));
	//def("learn_step_weight_decay",(void (*)(M&, M&, const float&, const float&)) learn_step_weight_decay<M>);
	//def("rprop",(void (*)(M&, M&, M&,M&, const float&)) rprop<M,M>);
}

template<class T>
void
export_transpose(){
	def("transpose", (void (*)(T&,const T&))transpose);
}

template<class V, class T, class I>
void
export_transposed_view(){
	typedef dense_matrix<V,T,row_major,I> M;
	typedef dense_matrix<V,T,column_major,I> N;
	def("transposed_view", (M*(*)(N&))transposed_view<V,T,I>,return_value_policy<manage_new_object, with_custodian_and_ward_postcall<1, 0> >());
	def("transposed_view", (N*(*)(M&))transposed_view<V,T,I>,return_value_policy<manage_new_object, with_custodian_and_ward_postcall<1, 0> >());
}

//template<class M>
//void
//export_multinomial_sampling(){
	//def("sample_multinomial",(void (*)(M&))sample_multinomial<typename M::value_type,typename M::memory_layout,typename M::memory_space_type, typename M::index_type>);
	//def("first_pool",	(void (*)(M&, M&, typename M::index_type))first_pooling<typename M::value_type,typename M::memory_layout,typename M::memory_space_type, typename M::index_type>);
	//def("first_pool_zeros",	(void (*)(M&,  typename M::index_type))first_pooling_zeros<typename M::value_type,typename M::memory_layout,typename M::memory_space_type, typename M::index_type>);
	//def("grid_to_matrix",    (void (*)(M&,M&,int))grid_to_matrix<typename M::value_type,typename M::memory_layout,typename M::memory_space_type, typename M::index_type>);
	//def("matrix_to_grid",    (void (*)(M&,M&,int))matrix_to_grid<typename M::value_type,typename M::memory_layout,typename M::memory_space_type, typename M::index_type>);
        //def("prob_max_pooling",    (void (*)(typename M::vec_type&,M&,int,bool))prob_max_pooling<typename M::value_type,typename M::memory_layout,typename M::memory_space_type, typename M::index_type>, (arg("sums"),arg("detection_layer"),arg("poolSize"),arg("sample")));
        //def("prob_max_pooling",    (void (*)(M&,int,bool))prob_max_pooling<typename M::value_type,typename M::memory_layout,typename M::memory_space_type, typename M::index_type>, (arg("detection_layer"),arg("poolSize"),arg("sample")));
//}


void export_matrix_ops(){
    enum_<cuv::reduce_functor>("reduce_functor")
        .value("ADD", RF_ADD)
        .value("ADD_SQUARED", RF_ADD_SQUARED)
        .value("MIN", RF_MIN)
        .value("MAX", RF_MAX)
        .value("ARGMAX", RF_ARGMAX)
        .value("ARGMIN", RF_ARGMIN)
        .value("LOGADDEXP", RF_LOGADDEXP)
        .value("ADDEXP", RF_ADDEXP)
        .value("MULT", RF_MULT)
        ;
	typedef dense_matrix<float,dev_memory_space,column_major> fdev;
	typedef dense_matrix<float,host_memory_space,column_major> fhost;
	typedef dense_matrix<float,host_memory_space,row_major> fhostr;
	typedef dense_matrix<float,dev_memory_space,row_major> fdevr;
	typedef dense_matrix<unsigned char,dev_memory_space,column_major> udev;
	typedef dense_matrix<unsigned char,host_memory_space,column_major> uhost;
	typedef dense_matrix<int,dev_memory_space,column_major> idev;
	typedef dense_matrix<int,host_memory_space,column_major> ihost;
	typedef dense_matrix<unsigned int,dev_memory_space,column_major> uidev;
	typedef dense_matrix<unsigned int,host_memory_space,column_major> uihost;
	typedef dense_matrix<unsigned int,dev_memory_space,row_major> uidevr;
	typedef dense_matrix<unsigned int,host_memory_space,row_major> uihostr;
	typedef dense_matrix<unsigned char,dev_memory_space,column_major> ucdev;
	typedef dense_matrix<unsigned char,host_memory_space,column_major> uchost;
	typedef dense_matrix<unsigned char,dev_memory_space,row_major> ucdevr;
	typedef dense_matrix<unsigned char,host_memory_space,row_major> uchostr;

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
	export_nullary_functor<uihost>();
	export_nullary_functor<uidev>();
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
	//export_binary_functor_simple<fhost,uhost>();
	//export_binary_functor_simple<fdev,udev>();

	export_reductions<fhost>();
	export_reductions<fdev>();
	export_reductions<fhostr>();
	export_reductions<fdevr>();

	export_reductions<uchost>();
	export_reductions<ucdev>();
	export_reductions<uchostr>();
	export_reductions<ucdevr>();

	//export_argmax_vec<float,host_memory_space>();
	//export_argmax_vec<float,dev_memory_space>();
	export_learn_step<fhost>();
	export_learn_step<fdev>();
	export_learn_step<fhostr>();
	export_learn_step<fdevr>();
	export_blas2<fdev,float>();
	export_blas2<fhost,float>();
	export_blas2<fhostr,float>();
	export_blas2<fdevr,float>();
	export_blas2<fdev,unsigned char>();
	export_blas2<fhost,unsigned char>();
	export_blas2<fhostr,unsigned char>();
	export_blas2<fdevr,unsigned char>();

	export_bitflip<fdev>();
	export_bitflip<fhost>();

	export_blockview<fdev>();
	export_blockview<fhost>();
	export_blockview<fdevr>();
	export_blockview<fhostr>();

	//export_pooling<fhostr>();
	//export_pooling<fdevr>();
	// transpose
	export_transpose<fhost>();
	export_transpose<fdev>();
	export_transpose<fhostr>();
	export_transpose<fdevr>();
	export_transposed_view<float,host_memory_space,unsigned int>();
	export_transposed_view<float,dev_memory_space,unsigned int>();

	//export_multinomial_sampling<dense_matrix<float,dev_memory_space,row_major> >();

}


