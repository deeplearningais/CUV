#include "rprop.hpp"

//#define sgn(a) (copysign(1.f,a))
#define sgn(a) ((a==0) ? 0 : copysign(1.f,a))

#define ETA_P 1.2f
#define ETA_M 0.5f
#define DELTA_MAX 50.0f
#define DELTA_MIN (1.0E-6)


template<class T, class O>
__global__ void rprop_kernel(T*W, T* dW, O* dW_old, T* rate, int n) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int off = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < n; i += off){
		O sn = (O)sgn(dW[i]);
		O s  = dW_old[i] * sn;
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
		dW_old[i] = (O)sgn(dwn);
		W[i]     += dwn;
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

	template<class V, class O, class I>
	void
	rprop_impl(dev_vector<V,I>& W, dev_vector<V,I>& dW, dev_vector<O,I>& dW_old, dev_vector<V,I>& rate){
		int num_threads = 512;
		int num_blocks  = min(512,(int)ceil((float)dW.size() / num_threads));
		rprop_kernel<<< num_threads, num_blocks>>>(W.ptr(), dW.ptr(), dW_old.ptr(), rate.ptr(), dW.size());
		cuvSafeCall(cudaThreadSynchronize());
	}

	template<class V, class O, class I>
	void
	rprop_impl(host_vector<V,I>& W, host_vector<V,I>& dW, host_vector<O,I>& dW_old, host_vector<V,I>& rate){
		for (unsigned int i = 0; i < dW.size(); i++){
			O sn = (O)sgn(dW[i]);
			O s  = dW_old[i] * sn;
			V dwn,rn=(V)0, r = rate[i];

			if (s > 0) {
				rn = min( ETA_P * r , DELTA_MAX );
				dwn = sn * rn;
			}
			else if (s < 0) {
				rn = max( ETA_M * r, DELTA_MIN);
				dwn = 0;
			}   
			else {
				dwn = sn * r;
			}
			/*__synchthreads();*/
			rate.set(i,rn);
			dW_old.set(i,(O)sgn(dwn));
			W.set(i, W[i]  + dwn);
		}
	}

	template<class __vector_type, class __old_vector_type>
	void rprop(__vector_type& W, __vector_type& dW, __old_vector_type& dW_old, __vector_type& rate){
		cuvAssert(dW.ptr());
		cuvAssert(dW_old.ptr());
		cuvAssert(rate.ptr());
		cuvAssert(dW.size() == dW_old.size());
		cuvAssert(dW.size() ==  rate.size());
		rprop_impl(W,dW,dW_old,rate);
	}



	template<class V, class I>
	void learn_step_weight_decay_impl(dev_vector<V,I>& W,dev_vector<V,I>& dW, const float& alpha, const float& beta){
		int num_threads = 512;
		int num_blocks  = min(512,(int)ceil((float)dW.size() / num_threads));
		learn_step_weight_decay_kernel<<< num_threads, num_blocks>>>(W.ptr(), dW.ptr(), alpha, beta, W.size());
		cuvSafeCall(cudaThreadSynchronize());
	}
	template<class V, class I>
	void learn_step_weight_decay_impl(host_vector<V,I>& W,host_vector<V,I>& dW, const float& alpha, const float& beta){
		for (unsigned int i = 0; i < W.size(); i++){
			W.set(i,alpha*dW[i] + beta*W[i]);
		}
	}
	template<class __vector_type>
	void learn_step_weight_decay(__vector_type& W,__vector_type& dW, const float& learnrate, const float& decay){
		cuvAssert(dW.ptr());
		cuvAssert(W.ptr());
		cuvAssert(W.size() == dW.size());
		learn_step_weight_decay_impl(W,dW,learnrate,1.f-learnrate*decay);
	}

#define RPROP_INSTANTIATE(V,O) \
	template void rprop( host_vector<V>&, host_vector<V>&, host_vector<O>&, host_vector<V>&); \
	template void rprop( dev_vector<V>&,  dev_vector<V>&, dev_vector<O>&, dev_vector<V>&);    
#define LSWD_INSTANTIATE(V) \
	template void learn_step_weight_decay( host_vector<V>&, host_vector<V>&, const float&,const float&); \
	template void learn_step_weight_decay( dev_vector<V>&,  dev_vector<V>&, const float&,const float&);

	RPROP_INSTANTIATE(float,float);
	RPROP_INSTANTIATE(float,signed char);
	LSWD_INSTANTIATE(float);

}
