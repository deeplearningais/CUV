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

namespace python_wrapping {
    /** 
     * Simple matrix multiplication that allocates a new tensor as the result.
     */
    template<class __value_type, class __memory_space_type, class __memory_layout_type>
        tensor<__value_type,__memory_space_type, __memory_layout_type>*
        prod(   const tensor<__value_type, __memory_space_type, __memory_layout_type>& A,
                const tensor<__value_type, __memory_space_type, __memory_layout_type>& B,
                char transA='n', char transB='n', const float& factAB=1.f){
            int shape[2];
            if (transA=='n')
                shape[0] = A.shape()[0];
            else
                shape[0] = A.shape()[1];
            if (transB=='n')
                shape[1] = B.shape()[1];
            else
                shape[1] = B.shape()[0];

            tensor<__value_type,__memory_space_type, __memory_layout_type>* C
                = new tensor<__value_type,__memory_space_type, __memory_layout_type>(extents[shape[0]][shape[1]]);
            prod(*C, A, B, transA, transB, factAB, 0.f);
            return C;
        }

    // convenience: create and return dst: if memory layout does not agree, default to row major
    template<class __value_type, class __memory_space_type, class __memory_layout_typeA, class __memory_layout_typeB>
        tensor<__value_type,__memory_space_type, row_major> *
        prod(   const tensor<__value_type,__memory_space_type, __memory_layout_typeA>& A,
                const tensor<__value_type,__memory_space_type,__memory_layout_typeB>& B,
                const float& factAB=1.f){
            tensor<__value_type,__memory_space_type,row_major>* C 
                = new tensor<__value_type,__memory_space_type, row_major>(extents[A.shape()[0]][B.shape()[1]]);
            cuv::prod(*C, A, B, factAB, 0.f);
            return C;
        }
    /** 
     * @brief Reduce a matrix to one row using specified reduce functor (or add them up by default). Return a new array.
     */
    template<class __value_type, class __value_type2, class __memory_space_type, class __memory_layout_type>
        tensor<__value_type, __memory_space_type>*
        reduce_to_row(const tensor<__value_type2, __memory_space_type, __memory_layout_type>& src,
                reduce_functor rf=RF_ADD, const __value_type2& factNew=1.f, const __value_type2& factOld=0.f){
            tensor<__value_type, __memory_space_type>*  dst = new tensor<__value_type, __memory_space_type>(src.shape()[1]);
            reduce_to_row(*dst, src, rf, factNew, factOld);
            return dst;

        }
    /** 
     * @brief Reduce a matrix to one column using specified reduce functor (or add them up by default). Return a new array.
     */
    template<class __value_type, class __value_type2, class __memory_space_type, class __memory_layout_type>
        tensor<__value_type, __memory_space_type>*
        reduce_to_col(const tensor<__value_type2, __memory_space_type, __memory_layout_type>& src,
                reduce_functor rf=RF_ADD, const __value_type2& factNew=1.f, const __value_type2& factOld=0.f){
            tensor<__value_type, __memory_space_type>*  dst = new tensor<__value_type, __memory_space_type>(src.shape()[0]);
            reduce_to_col(*dst, src, rf, factNew, factOld);
            return dst;

        }
}

template<class R>
void export_blas3() {
    // export matrix multiplication
    def("prod",(void (*)(R&,const R&,const R&,char, char, const float&, const float& ))
            prod<typename R::value_type,typename R::memory_space_type,typename R::memory_layout_type>,
            (arg("C"), arg("A"), arg("B"), arg("transA")='n', arg("transB")='n', arg("factAB")=1.f, arg("factC")=0.f));
    // convenience for use of layout instead of "n" and "t"
    typedef typename switch_memory_layout_type<R, typename other_memory_layout<typename R::memory_layout_type>::type >::type S;
    typedef typename switch_memory_layout_type<R, row_major >::type R_rm;
    def("prod",(void (*)(R&,const S&,const R&, const float&, const float& ))
            prod<typename R::value_type,typename R::memory_space_type,typename R::memory_layout_type>,
            (arg("C"), arg("A"), arg("B"), arg("factAB")=1.f, arg("factC")=0.f));
    def("prod",(void (*)(R&,const R&,const S&, const float&, const float& ))
            prod<typename R::value_type,typename R::memory_space_type,typename R::memory_layout_type>,
            (arg("C"), arg("A"), arg("B"), arg("factAB")=1.f, arg("factC")=0.f));
    // convenience prod that returns dst
    def("prod",(R* (*)(const R&,const R&,char, char, const float&))
            python_wrapping::prod<typename R::value_type,typename R::memory_space_type,typename R::memory_layout_type>,
            (arg("A"), arg("B"), arg("transA")='n', arg("transB")='n', arg("factAB")=1.f),
            return_value_policy<manage_new_object>());
    // convenience prod that returns dst
    def("prod",(R_rm* (*)(const R&,const S&, const float&))
            python_wrapping::prod<typename R::value_type,typename R::memory_space_type,typename R::memory_layout_type>,
            (arg("A"), arg("B"), arg("factAB")=1.f),
            return_value_policy<manage_new_object>());


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
    typedef typename switch_memory_layout_type<M, row_major>::type Vect;
    typedef typename switch_value_type<Vect, typename M::index_type>::type IndexVect;
    typedef typename switch_value_type<Vect, float>::type FloatVect;
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
    def("var", (float (*)(const M&)) var<value_type,typename M::memory_space_type>);
    def("sum",(Vect (*)(const M&, const int&)) sum<value_type,typename M::memory_space_type>,(arg("source"), arg("axis")));
}
template <class M>
void export_reductions_mv(){
    //typedef typename switch_value_type<M, typename M::index_type>::type::vec_type idx_vec;
    //typedef typename switch_value_type<M, float>::type::vec_type float_vec;
    typedef typename switch_memory_layout_type<M, row_major>::type Vect;
    typedef typename switch_value_type<Vect, typename M::index_type>::type IndexVect;
    typedef typename switch_value_type<Vect, float>::type FloatVect;
    typedef typename M::value_type value_type;
    typedef typename M::memory_space_type memory_space_type;
    typedef typename M::memory_layout_type memory_layout_type;
    def("reduce_to_col",(void (*) (Vect &, const M&, reduce_functor, const value_type &, const value_type &))
            reduce_to_col<value_type, value_type, memory_space_type, memory_layout_type>,
            (arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));

    def("reduce_to_row",(void (*) (Vect &, const M&, reduce_functor, const value_type &, const value_type &))
            reduce_to_row<value_type, value_type, memory_space_type, memory_layout_type>,
            (arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));

    def("reduce_to_col",(void (*) (IndexVect &, const M&, reduce_functor, const value_type &, const value_type &))
            reduce_to_col<typename M::index_type, value_type, memory_space_type, memory_layout_type>,
            (arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));

    def("reduce_to_row",(void (*) (IndexVect &, const M&, reduce_functor, const value_type &, const value_type &))
            reduce_to_row<typename M::index_type, value_type, memory_space_type, memory_layout_type>,
            (arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));

    def("reduce_to_col",(void (*) (FloatVect &, const M&, reduce_functor, const value_type &, const value_type &))
            reduce_to_col<float, value_type, memory_space_type, memory_layout_type>,
            (arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));

    def("reduce_to_row",(void (*) (FloatVect &, const M&, reduce_functor, const value_type &, const value_type &))
            reduce_to_row<float, value_type, memory_space_type, memory_layout_type>,
            (arg("vector"), arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f));


    def("reduce_to_row",(Vect* (*) (const M&, reduce_functor, const value_type &, const value_type &))
            python_wrapping::reduce_to_row<value_type, value_type, memory_space_type, memory_layout_type>,
            (arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f),
            return_value_policy<manage_new_object>());

    def("reduce_to_col",(Vect* (*) (const M&, reduce_functor, const value_type &, const value_type &))
            python_wrapping::reduce_to_col<value_type, value_type, memory_space_type, memory_layout_type>,
            (arg("matrix"),arg("reduce_functor")=RF_ADD,arg("factor_new")=(value_type)1.f,arg("factor_old")=(value_type)0.f),
            return_value_policy<manage_new_object>());
}

template <class M>
void export_blas2(){
    typedef typename M::value_type        V1;
    typedef typename M::memory_space_type M1;
    typedef typename M::memory_layout_type L1;
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

    def("learn_step_weight_decay",(void (*)(M&, const M&, const float&, const float&,const float&)) learn_step_weight_decay<typename M::value_type, typename M::memory_space_type>, (arg("W"),arg("dW"),arg("learnrate"),arg("l2decay")=0,arg("l1decay")=0));

    def("rprop", (void (*)(M&, M&, M&,  M&, const float&,const float&,const float&,const float&))rprop<V1,M1,V1>, (arg ("W"), arg ("dW"), arg ("dW_old"), arg ("learnrate") ,arg("l2cost")=0, arg("l1cost")=0, arg("eta_p")=1.2f, arg("eta_m")=0.5f));
    def("rprop", (void (*)(M&, M&, USM&,M&, const float&,const float&,const float&,const float&))rprop<V1,M1,signed char>, (arg ("W"), arg ("dW"), arg ("dW_old"), arg ("learnrate") ,arg("l2cost")=0,arg("l1cost")=0, arg("eta_p")=1.2f, arg("eta_m")=0.5f));
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

    export_blas3<fdev>();
    export_blas3<fhost>();
    export_blas3<fdevr>();
    export_blas3<fhostr>();
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

    export_reductions_mv<fhost>();
    export_reductions_mv<fdev>();
    export_reductions_mv<fhostr>();
    export_reductions_mv<fdevr>();

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


