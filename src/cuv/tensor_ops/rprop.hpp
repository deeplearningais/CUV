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





#ifndef __RPROP_HPP__
#define __RPROP_HPP__

#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/tensor.hpp>

namespace cuv{


    /**
     * @addtogroup blas1
     * @{
     */

	/** 
	 * @brief Does a gradient descent step using the "RPROP" algorithm.
	 * 
	 * @param W 	 Destination tensor
	 * @param dW	 Direction of gradient descent. Vector of same size as W. 
	 * @param dW_old Direction of gradient descent in privious step. Vector of same size as W. 
	 * @param rate	 Vector of same size as W containing separate learnrates for each entry. 
	 * @param decay  Scalar L2 weight decay (cost) parameter
	 * @param sparsedecay  Scalar L1 weight decay (cost) parameter
	 *
	 * 	Updates W according to the "RPROP" algorithm.
	 * 	Calculates W = (1-decay*rate)*W + rate * W
	 * 	where all multiplications are pointwise.
	 * 	Also rate and dW_old are updated at each step.
	 *
	 */
        template<class __value_type, class __memory_space_type, class S>
	void rprop(tensor<__value_type,__memory_space_type>& W, tensor<__value_type,__memory_space_type>& dW, tensor<S,__memory_space_type>& dW_old, tensor<__value_type,__memory_space_type>& rate, const float& decay = 0.0f, const float& sparsedecay=0.0f);

        /**
         * @overload
         *
         * casting column major to row major since working on linear memory anyway.
         */
        template<class __value_type, class __memory_space_type, class S>
	void rprop(tensor<__value_type,__memory_space_type, column_major>& W, tensor<__value_type,__memory_space_type, column_major>& dW, tensor<S,__memory_space_type, column_major>& dW_old, tensor<__value_type,__memory_space_type, column_major>& rate, const float& decay = 0.0f, const float& sparsedecay=0.0f){
            typedef tensor<__value_type, __memory_space_type> rm_tensor;
            typedef tensor<S, __memory_space_type> rm_tensor_S;
            rprop(*reinterpret_cast<rm_tensor*>(&W),*reinterpret_cast<rm_tensor*>(&dW),*reinterpret_cast<rm_tensor_S*>(&dW_old),*reinterpret_cast<rm_tensor*>(&rate),decay,sparsedecay);
        }
	/** 
	 * @brief Do a step of gradient descent with optional weight decay.
	 * 
	 * @param W 	Destination matrix
	 * @param dW	Direction of gradient descent. Vector of same size as W. 
	 * @param learnrate Scalar learnreate 
	 * @param decay	Scalar L2 weight decay (cost) parameter
	 * @param sparsedecay	Scalar L1 weight decay (cost) parameter
	 * 
	 * Calculates W = (1-decay*learnrate) * W + learnrate * dW
	 */
        template<class __value_type, class __memory_space_type>
	void learn_step_weight_decay(tensor<__value_type,__memory_space_type>& W, const tensor<__value_type,__memory_space_type>& dW, const float& learnrate, const float& decay = 0.0f, const float& sparsedecay=0.0f);

        /**
         * @overload
         *
         * casting column major to row major since working on linear memory anyway.
         */
        template<class __value_type, class __memory_space_type>
	void learn_step_weight_decay(tensor<__value_type,__memory_space_type, column_major>& W, const tensor<__value_type,__memory_space_type, column_major>& dW, const float& learnrate, const float& decay = 0.0f, const float& sparsedecay=0.0f){
            typedef tensor<__value_type, __memory_space_type> rm_tensor;
            learn_step_weight_decay(*reinterpret_cast<rm_tensor*>(&W),*reinterpret_cast<const rm_tensor*>(&dW),learnrate,decay,sparsedecay);
        }
        /** @} */ // blas1

}


#endif /* __RPROP_HPP__ */
