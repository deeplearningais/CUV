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
#define sgn(a) ((a==(typeof(a))0) ? 0.f : copysign(1.f,a))

#define DELTA_MAX 5.0f
#define DELTA_MIN (1.0E-8)

#ifdef __CDT_PARSER__
#define __global__
#endif

template<class T, class S>
__global__ void rprop_kernel(T*W, T* dW, S* dW_old, T* rate, int n, T decay, T sparsedecay, T eta_p, T eta_m) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int off = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < n; i += off){
                /*
                        for l1-norm, use ``Orthant-Wise Limited-memory Quasi-Newton Optimizer for L1-regularized Objectives''

			http://research.microsoft.com/en-us/downloads/b1eb1016-1738-4bd5-83a9-370c9d498a03/
                */

		T pg   = -dW[i]; // projected gradient
		T oldW = W[i];
		S sdW  = sgn(pg);
		pg    -= decay * oldW;

		S snW  = sgn(oldW);
		S tmp  = (snW==0) ? sgn(pg) : 0;
		pg    -= snW * sparsedecay;                  // if snW==0, apply to gradient instead...
		pg    -= tmp * min(sparsedecay, fabs(   pg));// ... keeping W at zero!

		S sn = (S)sgn(pg);
		S s  = dW_old[i] * sn;
		T delta=0, step=rate[i];

		if ( s > 0) {
			step = min( eta_p * step, DELTA_MAX);
			delta = sdW * step;
			if(sparsedecay!=0 && delta*pg<=(T)0) // we changed direction while projecting the gradient, don't execute step!
				delta = (T)0;
		}
		else if ( s < 0) {
			step = max( eta_m * step, DELTA_MIN);
			sdW  = 0;
		}
		else {
			if(sparsedecay==(T)0) // do not make a move when sparse decay is on (pg==0)
				delta = sn * step;
		}
		__syncthreads();
		rate[i]   = step;
		dW_old[i] = sdW;
		T newW    = oldW+delta;
		if (sparsedecay == 0.0f)
			W[i] = newW;
		else
			W[i] = (newW*oldW<(T)0) ? (T)0 : newW;
	}
} 

template<class T, class S>
__global__ void rrmsprop_kernel(T*W, T* dW, S* dW_old, T* rate, T* sW, int n, T avg_grad, T delta,  T decay, T sparsedecay, T eta_p, T eta_m, T delta_max, T delta_min) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int off = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < n; i += off){
                /*
                        for l1-norm, use ``Orthant-Wise Limited-memory Quasi-Newton Optimizer for L1-regularized Objectives''

			http://research.microsoft.com/en-us/downloads/b1eb1016-1738-4bd5-83a9-370c9d498a03/
                */

		T pg   = -dW[i]; // projected gradient
		T oldW = W[i];
		S sdW  = sgn(pg);
		pg    -= decay * oldW;

		S snW  = sgn(oldW);
		S tmp  = (snW==0) ? sgn(pg) : 0;
		pg    -= snW * sparsedecay;                  // if snW==0, apply to gradient instead...
		pg    -= tmp * min(sparsedecay, fabs(   pg));// ... keeping W at zero!

		S sn = (S)sgn(pg);
		S s  = dW_old[i] * sn;
		T d=0, step=rate[i];

		if ( s > 0) {
			step = min( eta_p * step, delta_max);
			d = sdW * step;
			if(sparsedecay!=0 && d*pg<=(T)0) // we changed direction while projecting the gradient, don't execute step!
				d = (T)0;
		}
		else if ( s < 0) {
			step = max( eta_m * step, delta_min);
			sdW  = 0;
		}
		else {
			if(sparsedecay==(T)0) // do not make a move when sparse decay is on (pg==0)
				d = sn * step;
		}
		__syncthreads();
		rate[i]   = step;
		dW_old[i] = sdW;
		sW[i]     = avg_grad * sW[i] + (1.f-avg_grad) * dW[i] * dW[i]; //pg*pg;
		T upd     = d / (sqrt(sW[i])+delta);
		T newW    = oldW+upd;
		if (sparsedecay == 0.0f)
			W[i] = newW;
		else
			W[i] = (newW*oldW<(T)0) ? (T)0 : newW;
	}
} 

template<class T>
__global__ void learn_step_weight_decay_kernel(T* A, const T* dA, T alpha, T beta, T sparsedecay, int n) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int off = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < n; i += off){
		A[i] = alpha*dA[i] + beta*A[i];
		/*T f  = alpha*dA[i] + beta*A[i];*/
		/*A[i] = f - sgn(f)*min(sparsedecay,fabs(f));*/
	}
}

template<class T>
__global__ void learn_step_weight_decay_momentum_kernel(T* A, T* M, const T* dA, T lr, T momentum_weight, T l2decay, T sparsedecay, int n) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int off = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < n; i += off){
        T m = M[i];
        m = momentum_weight * m - lr*(dA[i] + l2decay*A[i]);
        A[i] += m;
        M[i] = m;
		/*T f  = alpha*dA[i] + beta*A[i];*/
		/*A[i] = f - sgn(f)*min(sparsedecay,fabs(f));*/
	}
}



namespace cuv{

	template<class V, class S>
	void
	rprop_impl(tensor<V,dev_memory_space>& W, tensor<V,dev_memory_space>& dW, tensor<S,dev_memory_space>& dW_old, tensor<V,dev_memory_space>& rate, V decay, V sparsedecay, V eta_p, V eta_m){
		cuvAssert(decay >= 0);
		cuvAssert(sparsedecay >= 0);
		int num_threads = 512;
		int num_blocks  = min(512,(int)ceil((float)dW.size() / num_threads));
		rprop_kernel<<< num_blocks, num_threads>>>(W.ptr(), dW.ptr(), dW_old.ptr(), rate.ptr(), dW.size(), decay, sparsedecay, eta_p, eta_m);
		cuvSafeCall(cudaThreadSynchronize());
	}

	template<class T, class S>
	void
	rprop_impl(tensor<T,host_memory_space>& W, tensor<T,host_memory_space>& dW, tensor<S,host_memory_space>& dW_old, tensor<T,host_memory_space>& rate, T decay, T sparsedecay, T eta_p, T eta_m){
		cuvAssert(decay >=0);
		cuvAssert(sparsedecay >=0);
		for (unsigned int i = 0; i < dW.size(); i++){
			/*
			   for l1-norm, use ``Orthant-Wise Limited-memory Quasi-Newton Optimizer for L1-regularized Objectives''

				http://research.microsoft.com/en-us/downloads/b1eb1016-1738-4bd5-83a9-370c9d498a03/
			 */

			T pg   = -dW[i]; // projected gradient
			T oldW = W[i];
			S sdW  = sgn(pg);
			pg    -= decay * oldW;

			S snW  = sgn(oldW);
			S tmp  = (snW==0) ? sgn(pg) : 0;
			pg    -= snW * sparsedecay;                  // if snW==0, apply to gradient instead...
			pg    -= tmp * min(sparsedecay, fabs(   pg));// ... keeping W at zero!

			S sn = (S)sgn(pg);
			S s  = dW_old[i] * sn;
			T delta=0, step=rate[i];

			if ( s > 0) {
				step = min( eta_p * step, DELTA_MAX);
				delta = sdW * step;
				if(sparsedecay!=0 && delta*pg<=(T)0) // we changed direction while projecting the gradient, don't execute step!
					delta = (T)0;
			}
			else if ( s < 0) {
				step = max( eta_m * step, DELTA_MIN);
				sdW  = 0;
			}
			else {
				if(sparsedecay==(T)0) // do not make a move when sparse decay is on (pg==0)
					delta = sn * step;
			}
			rate[i]   = step;
			dW_old[i] = sdW;
			T newW    = oldW+delta;
			if (sparsedecay == 0.0f)
				W[i] = newW;
			else
				W[i] = (newW*oldW<(T)0) ? (T)0 : newW;
		}
	}

        template<class __value_type, class __memory_space_type, class S>
	void rprop(tensor<__value_type,__memory_space_type>& W, tensor<__value_type,__memory_space_type>& dW, tensor<S,__memory_space_type>& dW_old, tensor<__value_type,__memory_space_type>& rate, const float& decay, const float& sparsedecay, const float& eta_p, const float& eta_m){
		cuvAssert(dW.ptr());
		cuvAssert(dW_old.ptr());
		cuvAssert(rate.ptr());
		cuvAssert(dW.size() == dW_old.size());
		cuvAssert(dW.size() ==  rate.size());
		rprop_impl(W,dW,dW_old,rate,decay,sparsedecay, eta_p, eta_m);
	}

	template<class V, class S>
	void
	rrmsprop_impl(tensor<V,dev_memory_space>& W, tensor<V,dev_memory_space>& dW, tensor<S,dev_memory_space>& dW_old, tensor<V,dev_memory_space>& rate, tensor<V,dev_memory_space>& sW, V avg_grad, V delta, V decay, V sparsedecay, V eta_p, V eta_m, V delta_max, V delta_min){
		cuvAssert(decay >= 0);
		cuvAssert(sparsedecay >= 0);
		int num_threads = 512;
		int num_blocks  = min(512,(int)ceil((float)dW.size() / num_threads));
		rrmsprop_kernel<<< num_blocks, num_threads>>>(W.ptr(), dW.ptr(), dW_old.ptr(), rate.ptr(), sW.ptr(), dW.size(), avg_grad, delta, decay, sparsedecay, eta_p, eta_m, delta_max, delta_min);
// 		rprop_kernel<<< num_blocks, num_threads>>>(W.ptr(), dW.ptr(), dW_old.ptr(), rate.ptr(), dW.size(), decay, sparsedecay, eta_p, eta_m);
		cuvSafeCall(cudaThreadSynchronize());
	}

	template<class T, class S>
	void
	rrmsprop_impl(tensor<T,host_memory_space>& W, tensor<T,host_memory_space>& dW, tensor<S,host_memory_space>& dW_old, tensor<T,host_memory_space>& rate, tensor<T,host_memory_space>& sW, T avg_grad, T delta, T decay, T sparsedecay, T eta_p, T eta_m, T delta_max, T delta_min){
		cuvAssert(decay >=0);
		cuvAssert(sparsedecay >=0);
		for (unsigned int i = 0; i < dW.size(); i++){
			/*
			   for l1-norm, use ``Orthant-Wise Limited-memory Quasi-Newton Optimizer for L1-regularized Objectives''

				http://research.microsoft.com/en-us/downloads/b1eb1016-1738-4bd5-83a9-370c9d498a03/
			 */

			T pg   = -dW[i]; // projected gradient
			T oldW = W[i];
			S sdW  = sgn(pg);
			pg    -= decay * oldW;

			S snW  = sgn(oldW);
			S tmp  = (snW==0) ? sgn(pg) : 0;
			pg    -= snW * sparsedecay;                  // if snW==0, apply to gradient instead...
			pg    -= tmp * min(sparsedecay, fabs(   pg));// ... keeping W at zero!

			S sn = (S)sgn(pg);
			S s  = dW_old[i] * sn;
			T d=0, step=rate[i];

			if ( s > 0) {
				step = min( eta_p * step, delta_max);
				d = sdW * step;
				if(sparsedecay!=0 && d*pg<=(T)0) // we changed direction while projecting the gradient, don't execute step!
					d = (T)0;
			}
			else if ( s < 0) {
				step = max( eta_m * step, delta_min);
				sdW  = 0;
			}
			else {
				if(sparsedecay==(T)0) // do not make a move when sparse decay is on (pg==0)
					d = sn * step;
			}
			rate[i]   = step;
			dW_old[i] = sdW;
			sW[i]     = avg_grad * sW[i] + (1.f-avg_grad) * dW[i] * dW[i]; // pg*pg;
			T upd     = d / (sqrt(sW[i])+delta);
			T newW    = oldW+upd;
			if (sparsedecay == 0.0f)
				W[i] = newW;
			else
				W[i] = (newW*oldW<(T)0) ? (T)0 : newW;
		}
	}

	template<class __value_type, class __memory_space_type, class S>
	void rrmsprop(tensor<__value_type,__memory_space_type>& W, tensor<__value_type,__memory_space_type>& dW, tensor<S,__memory_space_type>& dW_old, tensor<__value_type,__memory_space_type>& rate, tensor<__value_type,__memory_space_type>& sW, const float& avg_grad, const float& delta, const float& decay, const float& sparsedecay, const float& eta_p, const float& eta_m, const float& delta_max, const float& delta_min){
		cuvAssert(dW.ptr());
		cuvAssert(dW_old.ptr());
		cuvAssert(rate.ptr());
		cuvAssert(sW.ptr());
		cuvAssert(dW.size() == dW_old.size());
		cuvAssert(dW.size() == rate.size());
		cuvAssert(dW.size() == sW.size());
		rrmsprop_impl(W,dW,dW_old,rate,sW,avg_grad,delta,decay,sparsedecay,eta_p,eta_m,delta_max,delta_min);
	}
	
	template<class V>
	void learn_step_weight_decay_impl(tensor<V,dev_memory_space>& W, const tensor<V,dev_memory_space>& dW, const float& alpha, const float& beta, const float& sparsedecay){
		int num_threads = 512;
		int num_blocks  = min(512,(int)ceil((float)dW.size() / num_threads));
		learn_step_weight_decay_kernel<<< num_blocks, num_threads>>>(W.ptr(), dW.ptr(), alpha, beta, sparsedecay, W.size());
		cuvSafeCall(cudaThreadSynchronize());
	}
	template<class V>
	void learn_step_weight_decay_impl(tensor<V,host_memory_space>& W, const tensor<V,host_memory_space>& dW, const float& alpha, const float& beta, const float& sparsedecay){
		const V* dwptr = dW.ptr();
		V* wptr  = W.ptr();
        const unsigned int size = W.size();
		for (unsigned int i = 0; i < size; i++){
			wptr[i]  = alpha*dwptr[i] + beta*wptr[i];
			/*wptr[i] -= sgn(wptr[i])* min(sparsedecay,fabs(wptr[i]));*/
		}
	}
        template<class __value_type, class __memory_space_type>
	void learn_step_weight_decay(tensor<__value_type,__memory_space_type>& W, const tensor<__value_type,__memory_space_type>& dW, const float& learnrate, const float& decay, const float& sparsedecay){
		cuvAssert(dW.ptr());
		cuvAssert(W.ptr());
		cuvAssert(W.size() == dW.size());
		learn_step_weight_decay_impl(W,dW,-learnrate,1.f-learnrate*decay,sparsedecay);
	}

	template<class V>
	void learn_step_weight_decay_momentum_impl(tensor<V,dev_memory_space>& W, tensor<V,dev_memory_space>& momentum, const tensor<V,dev_memory_space>& dW, const float& lr, const float& momentum_weight, const float& l2decay, const float& sparsedecay){
		int num_threads = 512;
		int num_blocks  = min(512,(int)ceil((float)dW.size() / num_threads));
		learn_step_weight_decay_momentum_kernel<<< num_blocks, num_threads >>>(W.ptr(), momentum.ptr(), dW.ptr(), lr, momentum_weight, l2decay, sparsedecay, W.size());
		cuvSafeCall(cudaThreadSynchronize());
	}
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
			/*wptr[i] -= sgn(wptr[i])* min(sparsedecay,fabs(wptr[i]));*/
		}
	}
    template<class V, class M>
	void learn_step_weight_decay_momentum(tensor<V,M>& W, tensor<V,M>& momentum, const tensor<V,M>& dW, const float& learnrate, const float& momentum_weight, const float& decay, const float& sparsedecay){
		cuvAssert(dW.ptr());
		cuvAssert(W.ptr());
		cuvAssert(W.size() == dW.size());
		cuvAssert(W.size() == momentum.size());
		learn_step_weight_decay_momentum_impl(W,momentum,dW,learnrate,momentum_weight,decay,sparsedecay);
	}

#define RPROP_INSTANTIATE(V,S) \
	template void rprop<V,host_memory_space,S>( tensor<V,host_memory_space>&, tensor<V,host_memory_space>&, tensor<S,host_memory_space>&, tensor<V,host_memory_space>&m, const float&, const float&, const float&, const float&); \
	template void rprop<V,dev_memory_space,S>( tensor<V,dev_memory_space>&,  tensor<V,dev_memory_space>&, tensor<S,dev_memory_space>&, tensor<V,dev_memory_space>&, const float&, const float&, const float&, const float&); \
   	template void rrmsprop<V,host_memory_space,S>( tensor<V,host_memory_space>&, tensor<V,host_memory_space>&, tensor<S,host_memory_space>&, tensor<V,host_memory_space>&, tensor<V,host_memory_space>&, const float&, const float&, const float&, const float&, const float&, const float&, const float&, const float&); \
   	template void rrmsprop<V,dev_memory_space,S>( tensor<V,dev_memory_space>&, tensor<V,dev_memory_space>&, tensor<S,dev_memory_space>&, tensor<V,dev_memory_space>&, tensor<V,dev_memory_space>&, const float&, const float&, const float&, const float&, const float&, const float&, const float&, const float&);
#define LSWD_INSTANTIATE(V) \
	template void learn_step_weight_decay( tensor<V,host_memory_space>&, const tensor<V,host_memory_space>&, const float&,const float&, const float&); \
	template void learn_step_weight_decay( tensor<V,dev_memory_space>&,  const tensor<V,dev_memory_space>&, const float&,const float&, const float&); \
	template void learn_step_weight_decay_momentum( tensor<V,host_memory_space>&, tensor<V,host_memory_space>&,const tensor<V,host_memory_space>&, const float&,const float&,const float&, const float&); \
	template void learn_step_weight_decay_momentum( tensor<V,dev_memory_space>&, tensor<V,dev_memory_space>&, const tensor<V,dev_memory_space>&, const float&,const float&,const float&, const float&);

	RPROP_INSTANTIATE(float,float);
	RPROP_INSTANTIATE(float,signed char);
	LSWD_INSTANTIATE(float);

}
