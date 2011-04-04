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





#include <cuv/tensor_ops/rprop.hpp>

//#define sgn(a) (copysign(1.f,a))
#define sgn(a) ((a==0) ? 0 : copysign(1.f,a))

#define ETA_P 1.2f
#define ETA_M 0.5f
#define DELTA_MAX 50.0f
#define DELTA_MIN (1.0E-8)

#ifdef __CDT_PARSER__
#define __global__
#endif

template<class T, class S>
__global__ void rprop_kernel(T*W, T* dW, S* dW_old, T* rate, int n, T decay) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int off = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < n; i += off){
		S sn = (S)sgn(dW[i] - decay*W[i]);
		S s  = dW_old[i] * sn;
		T dwn, rn=rate[i];

		if ( s > 0) {
			rn = min( ETA_P * rn, DELTA_MAX );
			dwn = sn * rn;
		}
		else if ( s < 0) {
			rn = max( ETA_M * rn, DELTA_MIN);
			dwn = 0;
		}   
		else {
			dwn = sn * rn;
		}   
		__syncthreads();
		rate[i]   = rn;
		dW_old[i] = (S)sgn(dwn);
		W[i]      = W[i] + dwn;
	}
} 


template<class T>
__global__ void learn_step_weight_decay_kernel(T* A, T* dA, T alpha, T beta, int n) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int off = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < n; i += off){
		A[i] = alpha*dA[i] + beta*A[i];
	}
}



namespace cuv{

	template<class V, class S>
	void
	rprop_impl(tensor<V,dev_memory_space>& W, tensor<V,dev_memory_space>& dW, tensor<S,dev_memory_space>& dW_old, tensor<V,dev_memory_space>& rate, V decay){
		cuvAssert(decay >= 0);
		int num_threads = 512;
		int num_blocks  = min(512,(int)ceil((float)dW.size() / num_threads));
		rprop_kernel<<< num_threads, num_blocks>>>(W.ptr(), dW.ptr(), dW_old.ptr(), rate.ptr(), dW.size(), decay);
		cuvSafeCall(cudaThreadSynchronize());
	}

	template<class V, class S>
	void
	rprop_impl(tensor<V,host_memory_space>& W, tensor<V,host_memory_space>& dW, tensor<S,host_memory_space>& dW_old, tensor<V,host_memory_space>& rate, V decay){
		cuvAssert(decay <1);
		cuvAssert(decay >=0);
		for (unsigned int i = 0; i < dW.size(); i++){
			S sn = (S)sgn(dW[i] - decay*W[i]);
			S s  = dW_old[i] * sn;
			V dwn,rn = rate[i];

			if (s > 0) {
				rn = min( ETA_P * rn , DELTA_MAX );
				dwn = sn * rn;
			}
			else if (s < 0) {
				rn = max( ETA_M * rn, DELTA_MIN);
				dwn = 0;
			}   
			else {
				dwn = sn * rn;
			}
			/*__synchthreads();*/
			rate[i]=rn;
			dW_old[i]=(S)sgn(dwn);
			W[i]=W[i] + dwn;
			
		}
	}

	template<class __tensor_type, class __old_tensor_type>
	void rprop(__tensor_type& W, __tensor_type& dW, __old_tensor_type& dW_old, __tensor_type& rate, const float &decay){
		cuvAssert(dW.ptr());
		cuvAssert(dW_old.ptr());
		cuvAssert(rate.ptr());
		cuvAssert(dW.size() == dW_old.size());
		cuvAssert(dW.size() ==  rate.size());
		rprop_impl(W,dW,dW_old,rate,decay);
	}



	template<class V>
	void learn_step_weight_decay_impl(tensor<V,dev_memory_space>& W,tensor<V,dev_memory_space>& dW, const float& alpha, const float& beta){
		int num_threads = 512;
		int num_blocks  = min(512,(int)ceil((float)dW.size() / num_threads));
		learn_step_weight_decay_kernel<<< num_threads, num_blocks>>>(W.ptr(), dW.ptr(), alpha, beta, W.size());
		cuvSafeCall(cudaThreadSynchronize());
	}
	template<class V>
	void learn_step_weight_decay_impl(tensor<V,host_memory_space>& W,tensor<V,host_memory_space>& dW, const float& alpha, const float& beta){
		for (unsigned int i = 0; i < W.size(); i++){
			W[i]=alpha*dW[i] + beta*W[i];
		}
	}
	template<class __tensor_type>
	void learn_step_weight_decay(__tensor_type& W,__tensor_type& dW, const float& learnrate, const float& decay){
		cuvAssert(dW.ptr());
		cuvAssert(W.ptr());
		cuvAssert(W.size() == dW.size());
		learn_step_weight_decay_impl(W,dW,learnrate,1.f-learnrate*decay);
	}

#define RPROP_INSTANTIATE(V,S) \
	template void rprop( tensor<V,host_memory_space>&, tensor<V,host_memory_space>&, tensor<S,host_memory_space>&, tensor<V,host_memory_space>&m, const float&); \
	template void rprop( tensor<V,dev_memory_space>&,  tensor<V,dev_memory_space>&, tensor<S,dev_memory_space>&, tensor<V,dev_memory_space>&, const float&);
#define LSWD_INSTANTIATE(V) \
	template void learn_step_weight_decay( tensor<V,host_memory_space>&, tensor<V,host_memory_space>&, const float&,const float&); \
	template void learn_step_weight_decay( tensor<V,dev_memory_space>&,  tensor<V,dev_memory_space>&, const float&,const float&);

	RPROP_INSTANTIATE(float,float);
	RPROP_INSTANTIATE(float,signed char);
	LSWD_INSTANTIATE(float);

}