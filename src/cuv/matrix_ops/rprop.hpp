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





#ifndef __MATRIX_RPROP_HPP__
#define __MATRIX_RPROP_HPP__

#include <cuv/basics/dense_matrix.hpp>
#include <cuv/tensor_ops/rprop.hpp>

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
template<class V, class M, class T, class I>
void learn_step_weight_decay(dense_matrix<V,M,T,I>& W, dense_matrix<V,M,T,I>& dW, const float& learnrate, const float& decay){
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
template<class V, class O, class M, class T, class I>
void rprop(dense_matrix<V,M,T,I>& W,
		   dense_matrix<V,M,T,I>& dW, 
		   dense_matrix<O,M,T,I>& dW_old,
		   dense_matrix<V,M,T,I>& rate,
		   const float& decay = 0.0f){ rprop(W.vec(),dW.vec(),dW_old.vec(), rate.vec(), decay);
}

}

#endif /* __MATRIX_RPROP_HPP__ */
