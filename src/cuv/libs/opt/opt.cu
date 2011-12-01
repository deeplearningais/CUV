#include <cuv/matrix_ops/matrix_ops.hpp>
#include "opt.hpp"

namespace cuv { namespace libs { namespace opt {



namespace impl{

        template<class V, class M, class L>
            void softmax_derivative(cuv::tensor<V, M, L>& dst, const cuv::tensor<V, M, L>& softmax_act, const cuv::tensor<V,M,L>& residual,  unsigned int vardim){
                typedef typename cuv::tensor<V, host_memory_space>::index_type index_type;

                const index_type n_variables = dst.shape(vardim);
                const index_type n_vals      = dst.shape(!vardim);

                if(dst.ptr()!=residual.ptr()){
                    dst += residual;
                }
                cuv::tensor<V,M>   red  (n_variables);
                cuv::tensor<V,M,L> prod (softmax_act.shape());
                cuv::apply_binary_functor(prod,softmax_act,residual,BF_MULT);
                if(vardim==1){
                    cuv::reduce_to_row  (red, prod,RF_ADD,  -1.f);
                    cuv::matrix_plus_row(dst, red);
                }
                else{
                    cuv::reduce_to_col(red, prod,RF_ADD, -1.f);
                    cuv::matrix_plus_col(dst, red);
                }

                dst *= softmax_act;
            }

    template<class V, class M, class L>
    void softmax(cuv::tensor<V, M,L>& dst, const cuv::tensor<V, M,L>& src, unsigned int vardim){
        typedef typename cuv::tensor<V, M, L>::index_type index_type;
        const index_type n_variables = dst.shape( vardim);

        cuv::tensor<V,M> red(cuv::extents[n_variables]);
        if(vardim==1) cuv::reduce_to_row(red, src, RF_LOGADDEXP);
        else          cuv::reduce_to_col(red, src, RF_LOGADDEXP);

        if(dst.ptr() != src.ptr()){
            dst = src;
        }
        if(vardim==1) cuv::matrix_plus_row(dst,-red);
        else          cuv::matrix_plus_col(dst,-red);
        cuv::apply_scalar_functor(dst,SF_EXP);
    }
}
    

template<class V, class M,class L>
void softmax_derivative(cuv::tensor<V, M,L>& dst, const cuv::tensor<V, M,L>& softmax_act, const cuv::tensor<V,M,L>& residual,unsigned int vardim){
    cuvAssert(equal_shape(dst,softmax_act));
    cuvAssert(equal_shape(dst,residual));
    cuvAssert(vardim == 0 || vardim==1);
    impl::softmax_derivative(dst,softmax_act,residual,vardim);
}

template<class V, class M, class L>
void softmax(cuv::tensor<V, M,L>& dst, const cuv::tensor<V, M,L>& src,unsigned int vardim){
    cuvAssert(equal_shape(dst,src));
    cuvAssert(vardim == 0 || vardim==1);
    impl::softmax(dst,src,vardim);
}

#define TENSOR(V,M,L) cuv::tensor<V,M,L>
#define INSTANTIATE(V,M,L) \
  template void softmax_derivative(TENSOR(V,M,L)&, const TENSOR(V,M,L)&, const TENSOR(V,M,L)&,unsigned int);\
  template void softmax(TENSOR(V,M,L)&, const TENSOR(V,M,L)&,unsigned int); 

INSTANTIATE(float,host_memory_space,row_major);
INSTANTIATE(float,host_memory_space,column_major);
INSTANTIATE(float,dev_memory_space,row_major);
            
} } }
