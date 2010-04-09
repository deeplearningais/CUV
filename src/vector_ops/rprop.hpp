#ifndef __RPROP_HPP__
#define __RPROP_HPP__

#include <tools/cuv_general.hpp>
#include <basics/vector.hpp>

namespace cuv{


	/** 
	 * @brief Does a gradient descent step using the "RPROP" algorithm.
	 * 
	 * @param W 	 Destination vector
	 * @param dW	 Direction of gradient descent. Vector of same size as W. 
	 * @param dW_old Direction of gradient descent in privious step. Vector of same size as W. 
	 * @param rate	 Vector of same size as W containing separate learnrates for each entry. 
	 * @param decay  Scalar weight decay (cost) parameter
	 *
	 * 	Updates W according to the "RPROP" algorithm.
	 * 	Calculates W = (1-decay*rate)*W + rate * W
	 * 	where all multiplications are pointwise.
	 * 	Also rate and dW_old are updated at each step.
	 *
	 */
	template<class __vector_type, class __old_vector_type>
	void rprop(__vector_type& W, __vector_type& dW, __old_vector_type& dW_old, __vector_type& rate, const float& decay = 0.0f);

	/** 
	 * @brief Do a step of gradient descent with optional weight decay.
	 * 
	 * @param W 	Destination matrix
	 * @param dW	Direction of gradient descent. Vector of same size as W. 
	 * @param learnrate Scalar learnreate 
	 * @param decay	Scalar weight decay (cost) parameter
	 * 
	 * Calculates W = (1-decay*learnrate) * W + learnrate * dW
	 */
	template<class __vector_type>
	void learn_step_weight_decay(__vector_type& W, __vector_type& dW, const float& learnrate, const float& decay = 0.0f);

}


#endif /* __RPROP_HPP__ */
