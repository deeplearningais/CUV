#include <cuv/matrix_ops/matrix_ops.hpp>
#include "opt.hpp"
#define sgn(a) ((a==(typeof(a))0) ? 0.f : copysign(1.f,a))

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
        if(vardim==1) cuv::reduce_to_row(red, src, RF_LOGADDEXP, -1.f);
        else          cuv::reduce_to_col(red, src, RF_LOGADDEXP, -1.f);

        if(dst.ptr() != src.ptr()){
            dst = src.copy();
        }
        if(vardim==1) cuv::matrix_plus_row(dst,red);
        else          cuv::matrix_plus_col(dst,red);
        cuv::apply_scalar_functor(dst,SF_EXP);
    }


    template<class T>
        __global__ void adagrad_kernel(T* Wptr, const T* dWptr, T* sWptr, T learnrate, T delta, T decay, T sparsedecay, unsigned int size) {
            const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int off = blockDim.x * gridDim.x;
            for (unsigned int i = idx; i < size; i += off){
                sWptr[i] += dWptr[i] * dWptr[i];
                float lr = learnrate / (sqrt(sWptr[i]) + delta);
                /*Wptr[i] = Wptr[i] - lr * (dWptr[i]);*/
                float f = Wptr[i] - lr * dWptr[i];
                Wptr[i] = sgn(f) * max(0.f, fabs(f) - learnrate * sparsedecay/lr);
            }
        }

    template<class V, class L>
        void adagrad(tensor<V,host_memory_space, L>& W, const tensor<V,host_memory_space, L>& dW, tensor<V,host_memory_space, L>& sW, const float& learnrate, const float& delta, const float& decay, const float& sparsedecay){
            unsigned int size = W.size();
            V* Wptr = W.ptr();
            const V* dWptr = dW.ptr();
            V* sWptr = sW.ptr();
            for(unsigned int i=0; i < size; i++){
                sWptr[i] += dWptr[i] * dWptr[i];
                float lr = learnrate / (sqrt(sWptr[i]) + delta);
                /*Wptr[i] = Wptr[i] - lr * (dWptr[i]);*/
                float f = Wptr[i] - lr * dWptr[i];
                Wptr[i] = sgn(f) * max(0.f, fabs(f) - learnrate * sparsedecay/lr);
            }
        }
    template<class V, class L>
        void adagrad(tensor<V,dev_memory_space,L>& W, const tensor<V,dev_memory_space,L>& dW, tensor<V,dev_memory_space,L>& sW, const float& learnrate, const float& delta, const float& decay, const float& sparsedecay){
            unsigned int size = dW.size();
            unsigned int num_threads = 512;
            unsigned int num_blocks  = min(512,(unsigned int)ceil((float)dW.size() / num_threads));
            adagrad_kernel<<< num_threads, num_blocks>>>(W.ptr(), dW.ptr(), sW.ptr(), learnrate,delta,decay,sparsedecay, size);
            cuvSafeCall(cudaThreadSynchronize());
        }

    template<class T>
        __global__ void rmsprop_kernel(T* Wptr, const T* dWptr, T* sWptr, T learnrate, T delta, T decay, T sparsedecay, unsigned int size, float grad_avg) {
            const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int off = blockDim.x * gridDim.x;
            for (unsigned int i = idx; i < size; i += off){
                sWptr[i] = grad_avg * sWptr[i] + (1.f-grad_avg) * dWptr[i] * dWptr[i];
                float lr = learnrate / (sqrt(sWptr[i]) + delta);
                /*Wptr[i] = Wptr[i] - lr * (dWptr[i]);*/
                float f = Wptr[i] - lr * dWptr[i];
                Wptr[i] = sgn(f) * max(0.f, fabs(f) - learnrate * sparsedecay/lr);
            }
        }

    template<class V, class L>
        void rmsprop(tensor<V,host_memory_space, L>& W, const tensor<V,host_memory_space, L>& dW, tensor<V,host_memory_space, L>& sW, const float& learnrate, const float& delta, const float& decay, const float& sparsedecay, const float& grad_avg){
            unsigned int size = W.size();
            V* Wptr = W.ptr();
            const V* dWptr = dW.ptr();
            V* sWptr = sW.ptr();
            for(unsigned int i=0; i < size; i++){
                sWptr[i] = grad_avg * sWptr[i] + (1.f-grad_avg) * dWptr[i] * dWptr[i];
                float lr = learnrate / (sqrt(sWptr[i]) + delta);
                /*Wptr[i] = Wptr[i] - lr * (dWptr[i]);*/
                float f = Wptr[i] - lr * dWptr[i];
                Wptr[i] = sgn(f) * max(0.f, fabs(f) - learnrate * sparsedecay/lr);
            }
        }
    template<class V, class L>
        void rmsprop(tensor<V,dev_memory_space,L>& W, const tensor<V,dev_memory_space,L>& dW, tensor<V,dev_memory_space,L>& sW, const float& learnrate, const float& delta, const float& decay, const float& sparsedecay, const float& grad_avg){
            unsigned int size = dW.size();
            unsigned int num_threads = 512;
            unsigned int num_blocks  = min(512,(unsigned int)ceil((float)dW.size() / num_threads));
            rmsprop_kernel<<< num_threads, num_blocks>>>(W.ptr(), dW.ptr(), sW.ptr(), learnrate,delta,decay,sparsedecay, size, grad_avg);
            cuvSafeCall(cudaThreadSynchronize());
        }
}
    
template<class V, class M, class L>
void adagrad(tensor<V,M,L>& W, const tensor<V,M,L>& dW, tensor<V,M,L>& sW, const float& learnrate, const float& delta, const float& decay, const float& sparsedecay){
    cuvAssert(equal_shape(W,dW));
    cuvAssert(equal_shape(W,sW));
    impl::adagrad(W,dW,sW,learnrate,delta,decay,sparsedecay);
}

template<class V, class M, class L>
void rmsprop(tensor<V,M,L>& W, const tensor<V,M,L>& dW, tensor<V,M,L>& sW, const float& learnrate, const float& delta, const float& decay, const float& sparsedecay, const float& grad_avg){
    cuvAssert(equal_shape(W,dW));
    cuvAssert(equal_shape(W,sW));
    impl::rmsprop(W,dW,sW,learnrate,delta,decay,sparsedecay,grad_avg);
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
  template void softmax(TENSOR(V,M,L)&, const TENSOR(V,M,L)&,unsigned int); \
  template void adagrad(TENSOR(V,M,L)&, const TENSOR(V,M,L)&,TENSOR(V,M,L)&,const float&, const float&, const float&, const float&); \
  template void rmsprop(TENSOR(V,M,L)&, const TENSOR(V,M,L)&,TENSOR(V,M,L)&,const float&, const float&, const float&, const float&, const float&); 

INSTANTIATE(float,host_memory_space,row_major);
INSTANTIATE(float,host_memory_space,column_major);
INSTANTIATE(float,dev_memory_space,row_major);
            
} } }
