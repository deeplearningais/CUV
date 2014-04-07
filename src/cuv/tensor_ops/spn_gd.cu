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



#include <cuv/tensor_ops/spn_gd.hpp>
#include <cuv/tensor_ops/functors.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>

#ifdef __CDT_PARSER__
#define __global__
#endif



namespace cuv{
template< class T>
__global__ void spn_gd_kernel(T*W, const T* dW, const T* dW_old, unsigned int n, float rate, const float decay, bool rescale, bool hard_bp,  int  n_size, unsigned int n_sub_size, float thresh) {
    bf_logaddexp<float> lae;     
    if ( (n_size > 0) && rescale) {
            extern __shared__ float tmp[];
            tmp[threadIdx.x] = -1E+38;

            unsigned int idx = blockIdx.x * n_sub_size + threadIdx.x;

            if ( threadIdx.x < n_sub_size ){
                T p_W = W[idx];
                T p_dW_old = dW_old[idx];
                T p_dW     = dW[idx];
                
                T delta;
                if (hard_bp){            
                    delta =  rate * ( p_dW_old - p_dW );
                } else {
                    delta =  rate *  (p_dW_old - p_dW);
                }
                //weight decay
	        if (hard_bp){
                    if (decay > 0){ 
			    delta -= rate*p_W*decay;
		    }
		    p_W += delta;
		} else{
                    if (decay > 0){ 
			    delta -= rate*expf(p_W)*decay;
		    }
	           float tmp = expf(p_W) + delta;
		   if (tmp <= 0)
		       p_W = -1E+38;
		   else
		       p_W = logf( tmp ); 
		}           
                //rescale weights ( project onto unit ball )                   
                tmp[threadIdx.x] = p_W; 
                
                //logarithmic sum 
                for ( unsigned int j = blockDim.x/2; j > 0; j/=2){
                    __syncthreads();
                    if (threadIdx.x < j){
                        tmp[threadIdx.x] = lae(tmp[threadIdx.x], tmp[threadIdx.x + j]);
                 //       tmp[threadIdx.x] = max(tmp[threadIdx.x], tmp[threadIdx.x + j]);
                    }
                }

                __syncthreads();
               
               if(hard_bp)
                   W[idx] = p_W - (tmp[0]/5.0); 
               else{ 
                   W[idx] = p_W - (tmp[0]);
	       }		   
                //reset shared memory of this thread
                tmp[threadIdx.x] = -1E+38;
            }
      } else {
        const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int off = blockDim.x * gridDim.x;
        for (unsigned int i = idx; i < n; i += off){

            T p_W = W[i];
            T p_dW_old = dW_old[i];
            T p_dW     = dW[i];
                        
            T delta;
            if ( hard_bp) {
                delta =  rate * ( p_dW_old - p_dW );
            } else {
                delta =  rate * ( p_dW_old - p_dW );    
            }
            //weight decay
            if (decay > 0) delta -= rate*p_W*decay;
            
            //sparse decay
            if (n_size > 0){ 
	        if (hard_bp)
		    W[i] += delta;
		else{
	           float tmp = expf(p_W) + delta;
		   if (tmp <= 0)
		       p_W = -1E+38;
		   else
		       W[i] = logf( tmp ); 
		}           
	    } else //convolutional layer is not in logspace
                W[i] += delta;
        }
    }  
}
    
    

template< class T>
void  spn_gd_host(T* W, const T* dW, const T* dW_old, unsigned int n, float rate, const float& decay, bool & rescale, bool hard_bp, int & n_size, unsigned int & n_sub_size, float thresh){   
    if ( (n_size > 0) && rescale) { // rescaling weights is only possible for spn layer..
        bf_logaddexp<float> lae;          
        for (unsigned int s = 0; s < n_size; s++){
              float sum = 0;
              for (unsigned int sub = 0; sub < n_sub_size; sub++){
                    bool first = true;                  
                    unsigned int i =  s * n_sub_size + sub;
                    T p_W = W[i];
                    T p_dW_old = dW_old[i];
                    T p_dW     = dW[i];
                    
                    T delta; 
                    if(hard_bp){
                        delta = rate * ( p_dW_old - p_dW );
                    } else {
                        delta = rate * ( p_dW_old - p_dW ) ;    
                    }
                    
                    //weight decay
                    if (decay > 0) delta -= rate*p_W*decay;
                    
                    //sparse decay
                    W[i] += delta;
                    if (rescale && (n_sub_size > 0)){
                        if ( first ){
                            sum = s * W[i];
                            first = false;
                        } else
                            sum = lae(sum, W[i]); // W[i] * 2
                    }
                }
                //rescale weights such that they sum up to one
                if (rescale && (n_sub_size > 0)){
                    for (unsigned int sub = 0; sub < n_sub_size; sub++){
                        unsigned int i =  s * n_sub_size + sub; 
                            sum = sum;
                            W[i] -= sum;
                        }  
                }
            }
    } else {
        for (unsigned int i = 0; i < n; i++){
                    T p_W = W[i];
                    T p_dW_old = dW_old[i];
                    T p_dW     = dW[i];
                    
                    T delta; 
                    if(hard_bp){
                        delta = rate * ( p_dW_old - p_dW );
                    } else {
                        delta = rate * ( p_dW_old - p_dW ) ;    
                    }
                    
                    //weight decay
                    if (decay > 0) delta -= rate*p_W*decay;
                    
                    //sparse decay
                    W[i] += delta;
                }
            }        
}
   

template<class V, class M>
void spn_gd(tensor<V,M>& W, const tensor<V,M>& dW, const tensor<V,M>& dW_old,  
                  bool hard_inference, bool rescale, float thresh, float rate,  const float & decay,  const float & sparsedecay){
        cuvAssert(dW.ptr());
        cuvAssert(dW_old.ptr());     
        cuvAssert(dW.size() == dW_old.size());
        cuvAssert(decay >= 0);
        cuvAssert(sparsedecay >= 0);
        
     
         int n_size;
         unsigned int n_sub_size;
         
         if ((W.shape().size() == 2) ){ // weights of any sum_layer
            n_size = W.shape(0);
            n_sub_size = W.shape(1);
        } else {
            n_size = -1;
            n_sub_size = 0;
        }
/*        
        std::cout << "rescaling, n_size = " << n_size << ", n_sub_size = " << n_sub_size << std::endl;
        for ( unsigned int i = 0;  i < W.shape().size(); i++)
            std::cout << "[" << W.shape(i) << "]";
        std::cout << std::endl;
*/           
        
        if(IsSame<M, host_memory_space>::Result::value){
            spn_gd_host (W.ptr(), dW.ptr(), dW_old.ptr(), dW.size(), rate, decay, rescale, hard_inference, n_size, n_sub_size, thresh);         
        }else{
            int num_threads = 512;
            int num_blocks  = (int)ceil((float)dW.size() / num_threads);
            if ( n_size > 0){ 
                num_blocks = n_size;
                //num_threads =  min( 256, (unsigned int) std::pow(2, ceil(log2f( (float)n_sub_size))));                
                num_threads =  min( 256, n_sub_size);                
            }
            unsigned int shared_mem = num_threads * sizeof(float);
            spn_gd_kernel<<< num_blocks, num_threads, shared_mem>>>(W.ptr(), dW.ptr(), dW_old.ptr(), dW.size(), rate, decay, rescale, hard_inference, n_size, n_sub_size, thresh);         

            cuvSafeCall(cudaThreadSynchronize());
        }    
    }


/*
 TODO?
template<class T>
__global__ void learn_step_weight_decay_kernel(T* A, const T* dA, T alpha, T beta, T sparsedecay, int n) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int off = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += off){
        A[i] = alpha*dA[i] + beta*A[i];
        //T f  = alpha*dA[i] + beta*A[i];
        //[i] = f - sgn(f)*min(sparsedecay,fabs(f));
    }
}


//TODO?
template<class T>
__global__ void learn_step_weight_decay_momentum_kernel(T* A, T* M, const T* dA, T lr, T momentum_weight, T l2decay, T sparsedecay, int n) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int off = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += off){
        T m = M[i];
        m = momentum_weight * m - lr*(dA[i] + l2decay*A[i]);
        A[i] += m;
        M[i] = m;
        // T f  = alpha*dA[i] + beta*A[i];
        // A[i] = f - sgn(f)*min(sparsedecay,fabs(f));
    }
}


    //TODO?
    template<class V>
    void learn_step_weight_decay_impl(tensor<V,dev_memory_space>& W, const tensor<V,dev_memory_space>& dW, const float& alpha, const float& beta, const float& sparsedecay){
        int num_threads = 512;
        int num_blocks  = min(512,(int)ceil((float)dW.size() / num_threads));
        learn_step_weight_decay_kernel<<< num_threads, num_blocks>>>(W.ptr(), dW.ptr(), alpha, beta, sparsedecay, W.size());
        cuvSafeCall(cudaThreadSynchronize());
    }
    //TODO?
    template<class V>
    void learn_step_weight_decay_impl(tensor<V,host_memory_space>& W, const tensor<V,host_memory_space>& dW, const float& alpha, const float& beta, const float& sparsedecay){
        const V* dwptr = dW.ptr();
        V* wptr  = W.ptr();
        const unsigned int size = W.size();
        for (unsigned int i = 0; i < size; i++){
            wptr[i]  = alpha*dwptr[i] + beta*wptr[i];
            //wptr[i] -= sgn(wptr[i])* min(sparsedecay,fabs(wptr[i]));
        }
    }
    //TODO?
        template<class __value_type, class __memory_space_type>
    void learn_step_weight_decay(tensor<__value_type,__memory_space_type>& W, const tensor<__value_type,__memory_space_type>& dW, const float& learnrate, const float& decay, const float& sparsedecay){
        cuvAssert(dW.ptr());
        cuvAssert(W.ptr());
        cuvAssert(W.size() == dW.size());
        learn_step_weight_decay_impl(W,dW,-learnrate,1.f-learnrate*decay,sparsedecay);
    }
    //TODO?
    template<class V>
    void learn_step_weight_decay_momentum_impl(tensor<V,dev_memory_space>& W, tensor<V,dev_memory_space>& momentum, const tensor<V,dev_memory_space>& dW, const float& lr, const float& momentum_weight, const float& l2decay, const float& sparsedecay){
        int num_threads = 512;
        int num_blocks  = min(512,(int)ceil((float)dW.size() / num_threads));
        learn_step_weight_decay_momentum_kernel<<< num_threads, num_blocks>>>(W.ptr(), momentum.ptr(), dW.ptr(), lr, momentum_weight, l2decay, sparsedecay, W.size());
        cuvSafeCall(cudaThreadSynchronize());
    }
    //TODO?
    template<class V>
    void learn_step_weight_decay_momentum_impl(tensor<V,host_memory_space>& W, tensor<V,host_memory_space>& momentum,const tensor<V,host_memory_space>& dW, const float& lr, const float& momentum_weight, const float& l2decay, const float& sparsedecay){
        const V* dwptr = dW.ptr();
        V* wptr  = W.ptr();
        V* mptr  = momentum.ptr();
        const unsigned int size = W.size();
        for (unsigned int i = 0; i < size; i++){
            float m = mptr[i];
            m  = momentum_weight * m - lr*(dwptr[i] - l2decay*wptr[i]);
            wptr[i] += m;
            mptr[i] = m;
            //wptr[i] -= sgn(wptr[i])* min(sparsedecay,fabs(wptr[i]));
        }
    }
    //TODO?
    template<class V, class M>
    void learn_step_weight_decay_momentum(tensor<V,M>& W, tensor<V,M>& momentum, const tensor<V,M>& dW, const float& learnrate, const float& momentum_weight, const float& decay, const float& sparsedecay){
        cuvAssert(dW.ptr());
        cuvAssert(W.ptr());
        cuvAssert(W.size() == dW.size());
        cuvAssert(W.size() == momentum.size());
        learn_step_weight_decay_momentum_impl(W,momentum,dW,learnrate,momentum_weight,decay,sparsedecay);
    }*/

#define  TENS(V,M)       tensor<V,M>
#define CTENS(V,M) const TENS(V,M)
#define SPN_GD_INSTANTIATE(V, M) \
    template void spn_gd <V, M> (TENS(V,M)&, CTENS(V,M)&, CTENS(V,M)&, bool, bool, float, float, const float&, const float&); 

    SPN_GD_INSTANTIATE(float, host_memory_space);
    SPN_GD_INSTANTIATE(float, dev_memory_space );

    //    LSWD_INSTANTIATE(float);
//#define LSWD_INSTANTIATE(V) \
//    template void learn_step_weight_decay( tensor<V,host_memory_space>&, const tensor<V,host_memory_space>&, const float&,const float&, const float&); \
//    template void learn_step_weight_decay( tensor<V,dev_memory_space>&,  const tensor<V,dev_memory_space>&, const float&,const float&, const float&); \
//    template void learn_step_weight_decay_momentum( tensor<V,host_memory_space>&, tensor<V,host_memory_space>&,const tensor<V,host_memory_space>&, const float&,const float&,const float&, const float&); \
//    template void learn_step_weight_decay_momentum( tensor<V,dev_memory_space>&, tensor<V,dev_memory_space>&, const tensor<V,dev_memory_space>&, const float&,const float&,const float&, const float&);

}
