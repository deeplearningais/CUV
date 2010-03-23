#ifndef __MATRIX_RPROP_HPP__
#define __MATRIX_RPROP_HPP__

#include <vector_ops/rprop.hpp>

namespace cuv{

	/*
	 * Wrappers for the vector-operation "RPROP"
	 */

template<class V, class M, class I>
void learn_step_weight_decay(dev_dense_matrix<V,M,I>& W, dev_dense_matrix<V,M,I>& dW, const float& learnrate, const float& decay){
	learn_step_weight_decay(W.vec(),dW.vec(),learnrate,decay);
}

template<class V, class M, class I>
void learn_step_weight_decay(host_dense_matrix<V,M,I>& W, host_dense_matrix<V,M,I>& dW, const float& learnrate, const float& decay){
	learn_step_weight_decay(W.vec(),dW.vec(),learnrate,decay);
}

template<class V, class O, class M, class I>
void rprop(dev_dense_matrix<V,M,I>& W,
		   dev_dense_matrix<V,M,I>& dW, 
		   dev_dense_matrix<O,M,I>& dW_old,
		   dev_dense_matrix<V,M,I>& rate,
		   const float& decay = 0.0f){ rprop(W.vec(),dW.vec(),dW_old.vec(), rate.vec(), decay);
}

template<class V, class O, class M, class I>
void rprop(host_dense_matrix<V,M,I>&  W,
		   host_dense_matrix<V,M,I>& dW, 
		   host_dense_matrix<O,M,I>& dW_old,
		   host_dense_matrix<V,M,I>& rate,
		   const float& decay = 0.0f){ rprop(W.vec(),dW.vec(),dW_old.vec(), rate.vec(), decay);
}
}

#endif /* __MATRIX_RPROP_HPP__ */
