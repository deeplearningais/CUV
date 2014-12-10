#include <boost/scoped_ptr.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include "opt.hpp"
#define sgn(a) ((a==(typeof(a))0) ? 0.f : copysign(1.f,a))
#define DIVUP(X, Y) (((X)%(Y)!=0) ? X/Y+1 : X/Y)

namespace cuv { namespace libs { namespace opt {

#define LOGREG_THREADS 128
#define LOGREG_GRAD_THREADS_X 128
#define LOGREG_GRAD_THREADS_Y 4


namespace impl{
    /**
      This is for patterns in the second dimension, and is more efficient.
      */
    template<class V, class V2>
    __global__ 
        void multinomial_logistic_loss_kernel(
                V* true_label_log_probs, V* correct_probs,
            const unsigned int n_patterns, const unsigned int n_labels,
            const V* probs, const V2* labels, const V* maxprobs){
            const int tidx = blockIdx.x * LOGREG_THREADS + threadIdx.x;
            if(tidx < n_patterns){
                const unsigned int label = labels[tidx];
                const float maxp = maxprobs[tidx];

                const float labelp = probs[label * n_patterns + tidx]; 

                true_label_log_probs[tidx] = __logf(labelp);
                if(labelp != maxp){
                    correct_probs[tidx] = 0;
                }else{
                    unsigned int n_max = 0;
                    for(unsigned int i=0; i<n_labels; i++){
                        n_max += probs[i * n_patterns + tidx] == maxp;
                    }
                    correct_probs[tidx] = 1.f / (float) n_max;
                }
            }
    }

    /**
      this is for patterns in the first dimension, and is inefficient.
      */
    template<class V, class V2>
    __global__ 
        void multinomial_logistic_loss_kernel_t(
                V* true_label_log_probs, V* correct_probs,
            const unsigned int n_patterns, const unsigned int n_labels,
            const V* probs, const V2* labels, const V* maxprobs){
            const int tidx = blockIdx.x * LOGREG_THREADS + threadIdx.x;
            if(tidx < n_patterns){
                const unsigned int label = labels[tidx];
                const float maxp = maxprobs[tidx];

                // TODO this is not coalesced!
                const float labelp = probs[tidx * n_labels + label]; 

                true_label_log_probs[tidx] = __logf(labelp);
                if(labelp != maxp){
                    correct_probs[tidx] = 0;
                }else{
                    unsigned int n_max = 0;
                    for(unsigned int i=0; i<n_labels; i++){
                        n_max += probs[tidx * n_labels + i] == maxp;
                    }
                    correct_probs[tidx] = 1.f / (float) n_max;
                }
            }
    }

    template<bool add, class V, class V2>
    __global__ 
        void multinomial_logistic_loss_grad_kernel(
                V* grads, const V* probs, const V2* labels, unsigned int n_patterns, unsigned int n_labels, float fact)
        {
            const unsigned int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
            const unsigned int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
            const unsigned int tidx = ty * n_patterns + tx;

            if(ty < n_labels && tx < n_patterns){
                const unsigned int label = labels[tx];
                float v = fact * ((label == ty) - probs[tidx]);
                if(add) 
                    grads[tidx] += v;
                else
                    grads[tidx] = v;
            }
        }

    /**
      transposed version.
      */
    template<bool add, class V, class V2>
    __global__ 
        void multinomial_logistic_loss_grad_kernel_t(
                V* grads, const V* probs, const V2* labels, 
                unsigned int n_patterns, unsigned int n_labels, float fact)
        {
            // note: X, Y swapped for transposed version
            const unsigned int tx = blockIdx.x * LOGREG_GRAD_THREADS_Y + threadIdx.x;
            const unsigned int ty = blockIdx.y * LOGREG_GRAD_THREADS_X + threadIdx.y;
            const unsigned int tidx = ty * n_labels + tx;

            if(ty < n_patterns && tx < n_labels){
                const unsigned int label = (unsigned int) (labels[ty] + 0.000001f);
                float v = fact * ((label == tx) - probs[tidx]);
                if(add) 
                    grads[tidx] += v;
                else
                    grads[tidx] = v;
            }
        }

        template<class V, class M, class L>
            void softmax_derivative(cuv::tensor<V, M, L>& dst, const cuv::tensor<V, M, L>& softmax_act, const cuv::tensor<V,M,L>& residual,  unsigned int vardim, float fact_old){
                typedef typename cuv::tensor<V, host_memory_space>::index_type index_type;

                const index_type n_variables = dst.shape(vardim);
                const index_type n_vals      = dst.shape(!vardim);
                
                boost::scoped_ptr<cuv::tensor<V,M,L> > tmp;
                if(fact_old != 0.f){
                    // remember previous value for end
                    tmp.reset(new cuv::tensor<V,M,L>(dst.copy()));
                }
                cuv::tensor<V,M>   red  (n_variables, dst.m_allocator);
                cuv::tensor<V,M,L> prod (softmax_act.shape(), dst.m_allocator);
                cuv::apply_binary_functor(prod,softmax_act,residual,BF_MULT);
                if(vardim==1){
                    cuv::reduce_to_row  (red, prod,RF_ADD,  -1.f);
                    cuv::matrix_op_vec(dst, residual, red, dst.ndim()-1, BF_ADD);
                }
                else{
                    cuv::reduce_to_col(red, prod,RF_ADD, -1.f);
                    cuv::matrix_op_vec(dst, residual, red, 0, BF_ADD);
                }

                dst *= softmax_act;

                if(tmp)
                    cuv::apply_binary_functor(dst, *tmp, BF_XPBY, fact_old);
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
                Wptr[i] = sgn(f) * max(0.f, fabs(f) - lr * sparsedecay);
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

    template<class T>
        __global__ void na_rmsprop(T* Wptr, const T* dWptr, T* oldWptr, T* sWptr, T* lrptr, T momentum, T grad_avg, T step_adapt, T delta, T lr_max, T lr_min, unsigned int size) {
            const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int off = blockDim.x * gridDim.x;
            for (unsigned int i = idx; i < size; i += off){
                sWptr[i] = grad_avg * sWptr[i] + (1.f-grad_avg) * dWptr[i] * dWptr[i];
                float upd = lrptr[i] * dWptr[i] / (sqrt(sWptr[i])+delta);
                float tmp = Wptr[i] - upd;
                float v = momentum*(tmp - oldWptr[i]);
                float f = tmp + v;
                Wptr[i] = sgn(f) * max(0.f, fabs(f) /*- learnrate * sparsedecay/lr*/);
                oldWptr[i] = tmp;
                float lr;
                if(sgn(v) == sgn(v + upd))
                    lr = lrptr[i] * (1 + step_adapt);
                else
                    lr = lrptr[i] * (1 - step_adapt);
                if(lr > lr_max)
                    lrptr[i] = lr_max;
                else if(lr < lr_min)
                    lrptr[i] = lr_min;
                else
                    lrptr[i] = lr;
            }
        }
    template<class V, class L>
        void na_rmsprop(tensor<V,host_memory_space, L>& W, const tensor<V,host_memory_space, L>& dW, tensor<V,host_memory_space, L>& oldW, tensor<V,host_memory_space, L>& sW, tensor<V,host_memory_space, L>& learnrates, const float& momentum, const float& grad_avg, const float& step_adapt, const float& delta, const float& lr_max, const float& lr_min){
            unsigned int size = W.size();
            V* Wptr = W.ptr();
            const V* dWptr = dW.ptr();
            V* oldWptr = oldW.ptr();
            V* sWptr = sW.ptr();
            V* lrptr = learnrates.ptr();
            for(unsigned int i=0; i < size; i++){
                sWptr[i] = grad_avg * sWptr[i] + (1.f-grad_avg) * dWptr[i] * dWptr[i];
                float upd = lrptr[i] * dWptr[i] / (sqrt(sWptr[i])+delta);
                float tmp = Wptr[i] - upd;
                float v = momentum*(tmp - oldWptr[i]);
                float f = tmp + v;
                Wptr[i] = sgn(f) * max(0.f, fabs(f) /*- learnrate * sparsedecay/lr*/);
                oldWptr[i] = tmp;
                float lr;
                if(sgn(v) == sgn(v + upd))
                    lr = lrptr[i] * (1 + step_adapt);
                else
                    lr = lrptr[i] * (1 - step_adapt);
                if(lr > lr_max)
                    lrptr[i] = lr_max;
                else if(lr < lr_min)
                    lrptr[i] = lr_min;
                else
                    lrptr[i] = lr;
            }
        }
    template<class V, class L>
        void na_rmsprop(tensor<V,dev_memory_space,L>& W, const tensor<V,dev_memory_space,L>& dW, tensor<V,dev_memory_space,L>& oldW, tensor<V,dev_memory_space,L>& sW, tensor<V,dev_memory_space,L>& learnrates, const float& momentum, const float& grad_avg, const float& step_adapt, const float& delta, const float& lr_max, const float& lr_min){
            unsigned int size = dW.size();
            unsigned int num_threads = 512;
            unsigned int num_blocks  = min(512,(unsigned int)ceil((float)dW.size() / num_threads));
            na_rmsprop<<< num_threads, num_blocks>>>(W.ptr(), dW.ptr(), oldW.ptr(), sW.ptr(), learnrates.ptr(), momentum, grad_avg, step_adapt, delta, lr_max, lr_min, size);
            cuvSafeCall(cudaThreadSynchronize());
        }
}

template<class V, class V2, class M, class L>
std::pair<float, float> multinomial_logistic_loss(
        cuv::tensor<V, M, L>& softmaxX, 
        const cuv::tensor<V, M, L>& X, 
        const cuv::tensor<V2, M, L>& Y, 
        int pattern_axis,
        boost::shared_ptr<allocator> alloc){

    int n_patterns = X.shape(pattern_axis);
    int n_labels = X.size() / n_patterns;

    cuvAssert(Y.ndim() == 1);
    cuvAssert(Y.shape(0) == n_patterns);

    // find maximum over columns
    tensor<V, M, L> red(cuv::extents[n_patterns], alloc);

    // determine softmax of X
    if(pattern_axis == 0)
        reduce_to_col(red, X, RF_MAX, -1.f, 0.f);
    else if(pattern_axis == X.ndim() - 1)
        reduce_to_row(red, X, RF_MAX, -1.f, 0.f);
    else{
        cuvAssert(false /* illegal dimension in multinomial_logistic_loss */);
    }
    matrix_op_vec(softmaxX, X, red, pattern_axis, BF_ADD);
    apply_scalar_functor(softmaxX, SF_EXP);
    if(pattern_axis == 0){
        reduce_to_col(red, softmaxX, RF_ADD);
    }else{
        reduce_to_row(red, softmaxX, RF_ADD);
    }
    matrix_op_vec(softmaxX, softmaxX, red, pattern_axis, BF_DIV);

    tensor<V, M, L> true_label_log_probs(n_patterns, alloc);
    tensor<V, M, L> correct_probs(n_patterns, alloc);

    if(pattern_axis == 0){
        reduce_to_col(red, softmaxX, RF_MAX);
    }else{
        reduce_to_row(red, softmaxX, RF_MAX);
    }
    dim3 threads(LOGREG_THREADS, 1);
    dim3 blocks(DIVUP(n_patterns, LOGREG_THREADS), 1);
    using namespace impl;
    if(pattern_axis == 0){
        // TODO this kernel is suboptimal!
        multinomial_logistic_loss_kernel_t<<<blocks, threads>>>(
                true_label_log_probs.ptr(), correct_probs.ptr(), 
                n_patterns, n_labels,
                softmaxX.ptr(), Y.ptr(), red.ptr()
                );
    }else{
        multinomial_logistic_loss_kernel<<<blocks, threads>>>(
                true_label_log_probs.ptr(), correct_probs.ptr(), 
                n_patterns, n_labels,
                softmaxX.ptr(), Y.ptr(), red.ptr()
                );
    }
    cuvSafeCall(cudaThreadSynchronize());

    std::pair<float, float> retval;
    retval.first = -cuv::sum(true_label_log_probs);
    retval.second = 1.f - cuv::mean(correct_probs);
    return retval;
}
template<class V, class V2, class M, class L>
void multinomial_logistic_loss_grad(
        cuv::tensor<V, M, L>& dmll_dX, 
        const cuv::tensor<V, M, L>& X, 
        const cuv::tensor<V2, M, L>& Y, 
        int pattern_axis, float fact_new, bool add
        ){
    int n_patterns = X.shape(pattern_axis);
    int n_labels = X.size() / n_patterns;
    cuvAssert(X.shape() == dmll_dX.shape());
    cuvAssert(Y.ndim() == 1);
    cuvAssert(Y.shape(0) == n_patterns);

    using namespace impl;
    if(pattern_axis == 0){
        // swapped X, Y for ``transposed'' kernel
        dim3 threads(LOGREG_GRAD_THREADS_Y, LOGREG_GRAD_THREADS_X);
        dim3 blocks(DIVUP(n_labels,   LOGREG_GRAD_THREADS_Y), 
                    DIVUP(n_patterns, LOGREG_GRAD_THREADS_X));
        if(!add){
            multinomial_logistic_loss_grad_kernel_t<false><<<blocks, threads>>>(dmll_dX.ptr(),
                    X.ptr(), Y.ptr(), n_patterns, n_labels, -1.f * fact_new);
        }else{
            multinomial_logistic_loss_grad_kernel_t<true><<<blocks, threads>>>(dmll_dX.ptr(),
                    X.ptr(), Y.ptr(), n_patterns, n_labels, -1.f * fact_new);
        }
    }else{
        dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
        dim3 blocks(DIVUP(n_patterns, LOGREG_GRAD_THREADS_X), 
                    DIVUP(n_labels,   LOGREG_GRAD_THREADS_Y));
        if(!add)
            multinomial_logistic_loss_grad_kernel<false><<<blocks, threads>>>(dmll_dX.ptr(),
                    X.ptr(), Y.ptr(), n_patterns, n_labels, -1.f * fact_new);
        else
            multinomial_logistic_loss_grad_kernel<true><<<blocks, threads>>>(dmll_dX.ptr(),
                    X.ptr(), Y.ptr(), n_patterns, n_labels, -1.f * fact_new);
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
void softmax_derivative(cuv::tensor<V, M,L>& dst, const cuv::tensor<V, M,L>& softmax_act, const cuv::tensor<V,M,L>& residual,unsigned int vardim, float fact_old){
    cuvAssert(equal_shape(dst,softmax_act));
    cuvAssert(equal_shape(dst,residual));
    cuvAssert(vardim == 0 || vardim==1);
    impl::softmax_derivative(dst,softmax_act,residual,vardim, fact_old);
}

template<class V, class M, class L>
void softmax(cuv::tensor<V, M,L>& dst, const cuv::tensor<V, M,L>& src,unsigned int vardim){
    cuvAssert(equal_shape(dst,src));
    cuvAssert(vardim == 0 || vardim==1);
    impl::softmax(dst,src,vardim);
}

template<class V, class M, class L>
void na_rmsprop(tensor<V,M,L>& W, const tensor<V,M,L>& dW, tensor<V,M,L>& oldW, tensor<V,M,L>& sW, tensor<V,M,L>& learnrates, const float& momentum, const float& grad_avg, const float& step_adapt, const float& delta, const float& lr_max, const float& lr_min){
    cuvAssert(equal_shape(W,dW));
    cuvAssert(equal_shape(W,oldW));
    cuvAssert(equal_shape(W,sW));
    cuvAssert(equal_shape(W,learnrates));
    impl::na_rmsprop(W,dW,oldW,sW,learnrates,momentum,grad_avg,step_adapt,delta,lr_max,lr_min);
}

#define TENSOR(V,M,L) cuv::tensor<V,M,L>
#define INSTANTIATE(V,M,L) \
  template void softmax_derivative(TENSOR(V,M,L)&, const TENSOR(V,M,L)&, const TENSOR(V,M,L)&,unsigned int,float);\
  template void softmax(TENSOR(V,M,L)&, const TENSOR(V,M,L)&,unsigned int); \
  template void adagrad(TENSOR(V,M,L)&, const TENSOR(V,M,L)&,TENSOR(V,M,L)&,const float&, const float&, const float&, const float&); \
  template void rmsprop(TENSOR(V,M,L)&, const TENSOR(V,M,L)&,TENSOR(V,M,L)&,const float&, const float&, const float&, const float&, const float&); \
  template void na_rmsprop(TENSOR(V,M,L)&,const TENSOR(V,M,L)&,TENSOR(V,M,L)&,TENSOR(V,M,L)&,TENSOR(V,M,L)&,const float&, const float&, const float&, const float&, const float&, const float&); 

#define INSTANTIATE_MLL(V,V2,M,L) \
  template std::pair<float, float> multinomial_logistic_loss(TENSOR(V,M,L)&, const TENSOR(V,M,L)&, const TENSOR(V2,M,L)&, int pattern_axis, boost::shared_ptr<allocator>);\
  template void multinomial_logistic_loss_grad(TENSOR(V,M,L)&, const TENSOR(V,M,L)&, const TENSOR(V2,M,L)&, int pattern_axis, float, bool add);\

INSTANTIATE(float,host_memory_space,row_major);
INSTANTIATE(float,host_memory_space,column_major);
INSTANTIATE(float,dev_memory_space,row_major);

INSTANTIATE_MLL(float,float,dev_memory_space,row_major);
INSTANTIATE_MLL(float,unsigned int,dev_memory_space,row_major);
            
} } }
