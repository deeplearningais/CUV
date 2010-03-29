#ifndef __MATRIX_RPROP_HPP__
#define __MATRIX_RPROP_HPP__

#include <vector_ops/rprop.hpp>

namespace cuv{


	/** 
	 * @brief Do a step of gradient descent with optional weight decay.
	 * 
	 * @param W 	Destination matrix
	 * @param dW	Direction of gradient descent. Matrix of same size as W. 
	 * @param learnrate Scalar learnreate 
	 * @param decay	Scalar weight decay (cost) parameter
	 * 
	 * Calculates W = (1-decay*learnrate) * W + learnrate * dW
	 */
template<class V, class M, class I>
void learn_step_weight_decay(dev_dense_matrix<V,M,I>& W, dev_dense_matrix<V,M,I>& dW, const float& learnrate, const float& decay){
	learn_step_weight_decay(W.vec(),dW.vec(),learnrate,decay);
}

  /**
   * @copydoc learn_step_weight_decay(dev_dense_matrix<V,M,I>& W, dev_dense_matrix<V,M,I>& dW, const float& learnrate, const float& decay)
   */
template<class V, class M, class I>
void learn_step_weight_decay(host_dense_matrix<V,M,I>& W, host_dense_matrix<V,M,I>& dW, const float& learnrate, const float& decay){
	learn_step_weight_decay(W.vec(),dW.vec(),learnrate,decay);
}

	/*
	 * Wrappers for the vector-operation "RPROP"
	 */
/** 
 * @brief Does a gradient descent step using the "RPROP" algorithm.
 * 
 * @param W 	 Destination matrix
 * @param dW	 Direction of gradient descent. Matrix of same size as W. 
 * @param dW_old Direction of gradient descent in privious step. Matrix of same size as W. 
 * @param rate	 Matrix of same size as W containing separate learnrates for each entry. 
 * @param decay  Scalar weight decay (cost) parameter
 *
 * 	Updates W according to the "RPROP" algorithm.
 * 	Calculates W = (1-decay*rate)*W + rate * W
 * 	where all multiplications are pointwise.
 * 	Also rate and dW_old are updated at each step.
 *
 */
template<class V, class O, class M, class I>
void rprop(dev_dense_matrix<V,M,I>& W,
		   dev_dense_matrix<V,M,I>& dW, 
		   dev_dense_matrix<O,M,I>& dW_old,
		   dev_dense_matrix<V,M,I>& rate,
		   const float& decay = 0.0f){ rprop(W.vec(),dW.vec(),dW_old.vec(), rate.vec(), decay);
}

/**
* @copydoc rprop(dev_dense_matrix<V,M,I>& W, dev_dense_matrix<V,M,I>& dW, dev_dense_matrix<O,M,I>& dW_old, dev_dense_matrix<V,M,I>& rate, const float& decay = 0.0f)
*/
template<class V, class O, class M, class I>
void rprop(host_dense_matrix<V,M,I>&  W,
		   host_dense_matrix<V,M,I>& dW, 
		   host_dense_matrix<O,M,I>& dW_old,
		   host_dense_matrix<V,M,I>& rate,
		   const float& decay = 0.0f){ rprop(W.vec(),dW.vec(),dW_old.vec(), rate.vec(), decay);
}
}

#endif /* __MATRIX_RPROP_HPP__ */
