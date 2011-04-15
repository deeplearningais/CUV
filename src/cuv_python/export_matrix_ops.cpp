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

#include <cuv/basics/tensor.hpp>
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

template<class R, class S, class T>
void export_blas3() {
	def("prod",(void (*)(R&,const S&,const T&,char, char, const float&, const float& ))prod<typename R::value_type,typename R::memory_space_type,typename R::memory_layout_type>, (
				arg("C"), arg("A"), arg("B"), arg("transA")='n', arg("transB")='n', arg("factAB")=1.f, arg("factC")=0.f
				));
}

template<class M>
void export_nullary_functor() {
	def("apply_nullary_functor",
	   (void (*)(M&,const NullaryFunctor&)) 
	   apply_0ary_functor< typename M::value_type, typename M::memory_space_type>);
	def("apply_nullary_functor",
	   (void (*)(M&,const NullaryFunctor&, const typename M::value_type&)) 
	   apply_0ary_functor< typename M::value_type, typename M::memory_space_type>);

	// convenience wrappers
	def("sequence", (void (*)(M&)) sequence);
	def("fill",     (void (*)(M&,const typename M::value_type&)) fill);
}

template<class M>
void export_scalar_functor() {
	typedef tensor<unsigned char, typename M::memory_space_type, typename M::memory_layout_type> mask_t;
	// in place
	//def("apply_scalar_functor",
	   //(void (*)(M&,const ScalarFunctor&)) 
	   //apply_scalar_functor< typename M::value_type, typename M::memory_space_type >);
	//def("apply_scalar_functor",
	   //(void (*)(M&,const ScalarFunctor&, const typename M::value_type&)) 
	   //apply_scalar_functor< typename M::value_type, typename M::memory_space_type, typename M::value_type>);
        def("apply_scalar_functor",
           (void (*)(M&,const ScalarFunctor&, const mask_t*)) 
           apply_scalar_functor<M>, (arg("src/dst"),arg("functor"),arg("mask")=object()));
        def("apply_scalar_functor",
           (void (*)(M&,const ScalarFunctor&, const typename M::value_type&, const mask_t*)) 
           apply_scalar_functor<M>, (arg("src/dst"),arg("functor"),arg("functor argument"),arg("mask")=object()));
        // not in place
        def("apply_scalar_functor",
           (void (*)(M&, const M&, const ScalarFunctor&, const mask_t*)) 
           apply_scalar_functor<M>, (arg("dst"),arg("src"),arg("functor"),arg("mask")=object()));
        def("apply_scalar_functor",
           (void (*)(M&,const M&,const ScalarFunctor&, const typename M::value_type&, const mask_t*)) 
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

template <class M>
void export_reductions(){
	//typedef typename switch_value_type<M, typename M::index_type>::type::vec_type idx_vec;
	//typedef typename switch_value_type<M, float>::type::vec_type float_vec;
	typedef typename M::value_type value_type;
	typedef typename M::memory_space_type memory_space_type;
	typedef typename M::memory_layout_type memory_layout_type;
	def("has_inf",(bool (*)(const M&)) has_inf<value_type,typename M::memory_space_type>);
	def("has_nan",(bool (*)(const M&)) has_nan<value_type,typename M::memory_space_type>);
	def("sum",(float (*)(const M&)) sum<value_type,typename M::memory_space_type>);
	def("norm1",(float (*)(const M&)) norm1<value_type,typename M::memory_space_type>);
	def("norm2",(float (*)(const M&)) norm2<value_type,typename M::memory_space_type>);
	def("maximum",(float (*)(const M&)) maximum<value_type,typename M::memory_space_type>);
	def("minimum",(float (*)(const M&)) minimum<value_type,typename M::memory_space_type>);
	def("mean", (float (*)(const M&)) mean<value_type,typename M::memory_space_type>);
        def("reduce_to_col", reduce_to_col<value_type, value_type, memory_space_type, memory_layout_type>,(arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));
        def("reduce_to_row", reduce_to_row<value_type, value_type, memory_space_type, memory_layout_type>,(arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));
        def("reduce_to_col", reduce_to_col<typename M::index_type, value_type, memory_space_type, memory_layout_type>,(arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));
        def("reduce_to_row", reduce_to_row<typename M::index_type, value_type, memory_space_type, memory_layout_type>,(arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));
        def("reduce_to_col", reduce_to_col<float, value_type, memory_space_type, memory_layout_type>,(arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));
        def("reduce_to_row", reduce_to_row<float, value_type, memory_space_type, memory_layout_type>,(arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));
}

template <class M>
void export_blas2(){
	typedef typename M::value_type        V1;
	typedef typename M::memory_space_type M1;
	typedef typename M::memory_layout_type L1;
	typedef typename switch_memory_layout_type<M,row_major>::type VECT;
	def("matrix_plus_col", matrix_plus_col<V1,M1,L1>);
	def("matrix_times_col", matrix_times_col<V1,M1,L1>);
	def("matrix_divide_col", matrix_divide_col<V1,M1,L1>);
	def("matrix_plus_row", matrix_plus_row<V1,M1,L1>);
	def("matrix_times_row", matrix_times_row<V1,M1,L1>);
	def("matrix_divide_row", matrix_divide_row<V1,M1,L1>);
}

template <class M>
void export_blockview(){
	typedef typename M::index_type I;
	def("blockview",(M*(*)(M&,I,I,I,I))
			blockview<typename M::value_type,typename M::memory_space_type, typename M::memory_layout_type,typename M::index_type>,return_value_policy<manage_new_object>());
}

template <class M>
void export_learn_step(){
	typedef typename M::value_type        V1;
	typedef typename M::memory_space_type M1;
	typedef typename M::memory_layout_type L1;

	typedef typename switch_value_type<M,signed char>::type USM;

	def("learn_step_weight_decay",(void (*)(M&, M&, const float&, const float&)) learn_step_weight_decay<typename M::value_type, typename M::memory_space_type>);

	def("rprop", (void (*)(M&, M&, M&,  M&, const float&))rprop<V1,M1,V1>, (arg ("W"), arg ("dW"), arg ("dW_old"), arg ("learnrate") ,arg("cost")=0));
	def("rprop", (void (*)(M&, M&, USM&,M&, const float&))rprop<V1,M1,signed char>, (arg ("W"), arg ("dW"), arg ("dW_old"), arg ("learnrate") ,arg("cost")=0));
}

template<class T>
void
export_transpose(){
	def("transpose", (void (*)(T&,const T&))transpose);
}

template<class V, class T>
void
export_transposed_view(){
	typedef tensor<V,T,row_major> M;
	typedef tensor<V,T,column_major> N;
	def("transposed_view", (M*(*)(N&))transposed_view_p<V,T>,return_value_policy<manage_new_object, with_custodian_and_ward_postcall<1, 0> >());
	def("transposed_view", (N*(*)(M&))transposed_view_p<V,T>,return_value_policy<manage_new_object, with_custodian_and_ward_postcall<1, 0> >());
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
	typedef tensor<float,dev_memory_space,column_major> fdev;
	typedef tensor<float,host_memory_space,column_major> fhost;
	typedef tensor<float,host_memory_space,row_major> fhostr;
	typedef tensor<float,dev_memory_space,row_major> fdevr;
	typedef tensor<unsigned char,dev_memory_space,column_major> udev;
	typedef tensor<unsigned char,host_memory_space,column_major> uhost;
	typedef tensor<int,dev_memory_space,column_major> idev;
	typedef tensor<int,host_memory_space,column_major> ihost;
	typedef tensor<unsigned int,dev_memory_space,column_major> uidev;
	typedef tensor<unsigned int,host_memory_space,column_major> uihost;
	typedef tensor<unsigned int,dev_memory_space,row_major> uidevr;
	typedef tensor<unsigned int,host_memory_space,row_major> uihostr;
	typedef tensor<unsigned char,dev_memory_space,column_major> ucdev;
	typedef tensor<unsigned char,host_memory_space,column_major> uchost;
	typedef tensor<unsigned char,dev_memory_space,row_major> ucdevr;
	typedef tensor<unsigned char,host_memory_space,row_major> uchostr;

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

	export_learn_step<fhost>();
	export_learn_step<fdev>();
	export_learn_step<fhostr>();
	export_learn_step<fdevr>();
	export_blas2<fdev>();
	export_blas2<fhost>();
	export_blas2<fhostr>();
	export_blas2<fdevr>();
	export_blas2<fdev>();
	export_blas2<fhost>();
	export_blas2<fhostr>();
	export_blas2<fdevr>();


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
	export_transposed_view<float,host_memory_space>();
	export_transposed_view<float,dev_memory_space>();

	//export_multinomial_sampling<tensor<float,dev_memory_space,row_major> >();

}


