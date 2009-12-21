#include "rprop.hpp"

#define sgn(a) (copysign(1.f,a))

#define ETA_P 1.2f
#define ETA_M 0.5f
#define DELTA_MAX 50.0f
#define DELTA_MIN (1.0E-6)


template<class T, class O>
__global__ void rprop_kernel(T* dW, O* dW_old, T* rate, int n) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int off = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < n; i += off){
		if (i >= n) 
			return;

		O sn = sgn(dW[i]);
		O s  = dW_old[i] * sn;
		dW_old[i] = sn;

		if ( s > 0) {
			rate[i] = min( ETA_P * rate[i] , DELTA_MAX );
			dW[i] = sgn(dW[i]) * rate[i];
		}
		else if ( s < 0) {
			rate[i] = max( ETA_M * rate[i], DELTA_MIN);
			dW[i] = 0;
		}   
		else {
			dW[i] = sgn(dW[i]) * rate[i];
		}   
	}
} 


namespace cuv{

	template<class V, class O, class I>
	void
	rprop_impl(dev_vector<V,I>& dW, dev_vector<O,I>& dW_old, dev_vector<V,I>& rate){
		int num_threads = 512;
		int num_blocks  = min(512,(int)ceil((float)dW.size() / num_threads));
		rprop_kernel<<< num_threads, num_blocks>>>(dW.ptr(), dW_old.ptr(), rate.ptr(), dW.size());
		cuvSafeCall(cudaThreadSynchronize());
	}

	template<class V, class O, class I>
	void
	rprop_impl(host_vector<V,I>& dW, host_vector<O,I>& dW_old, host_vector<V,I>& rate){
		for (unsigned int i = 0; i < dW.size(); i++){
			O sn = sgn(dW[i]);
			O s  = dW_old[i] * sn;
			dW_old[i] = sn;

			if (s > 0) {
				rate[i] = min( ETA_P * rate[i] , DELTA_MAX );
				dW[i] = sgn(dW[i]) * rate[i];
			}
			else if (s < 0) {
				rate[i] = max( ETA_M * rate[i], DELTA_MIN);
				dW[i] = 0;
			}   
			else {
				dW[i] = sgn(dW[i]) * rate[i];
			}
		}
	}

	template<class __vector_type, class __old_vector_type>
	void rprop(__vector_type& dW, __old_vector_type& dW_old, __vector_type& rate){
		cuvAssert(dW.ptr());
		cuvAssert(dW_old.ptr());
		cuvAssert(rate.ptr());
		cuvAssert(dW.size() == dW_old.size());
		cuvAssert(dW.size() ==  rate.size());
		rprop_impl(dW,dW_old,rate);
	}

#define RPROP_INSTANTIATE(V,O) \
	template void rprop( host_vector<V>&, host_vector<O>&, host_vector<V>&); \
	template void rprop( dev_vector<V>&, dev_vector<O>&, dev_vector<V>&);

	RPROP_INSTANTIATE(float,float);
	RPROP_INSTANTIATE(float,signed char);

}
