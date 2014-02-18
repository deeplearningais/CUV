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

#ifndef __CUV_OPT_HPP__
#define __CUV_OPT_HPP__
#include<cuv/basics/tensor.hpp>

namespace cuv{ namespace libs{
	/// function optimization
	namespace opt
	{
		/**
		 * @addtogroup libs
		 * @{
		 * @defgroup opt Function Optimization
		 * @{
		 */

		/**
		 * calculate derivative of softmax.
         *
         * Calculates the SoftMax function \f$S(\vec x) = exp(x_i)/Sum_k(exp(x_k))\f$
         * for \f$m\f$ multinomial variables with \f$n\f$ values.
         *
         * @warning this /adds/ to the values already in dst, so you may need to zero dst first!
         *
		 * @param dst     the value of \f$ S(\vec x) \f$ of size \f$ n\times m\f$
		 * @param src     the input values to be softmaxed
         * @param vardim  the dimension in which the variables are stored
		 */
		template<class V, class M, class L>
		void softmax(cuv::tensor<V, M,L>& dst, const cuv::tensor<V, M,L>& src, unsigned int vardim=1);

		/**
		 * calculate derivative of softmax.
         *
         * Calculates the derivative of SoftMax function \f$S(\vec x) = exp(x_i)/Sum_k(exp(x_k))\f$
         * for \f$m\f$ multinomial variables with \f$n\f$ values.
         *
         * @warning this /adds/ to the values already in dst, so you may need to zero dst first!
         *
		 * @param dst         destination tensor of size \f$ n\times m \f$
		 * @param softmax_act the value of \f$ S(\vec x) \f$ of size \f$ n\times m\f$
         * @param residual    the residual of size \f$ S(\vec x) \f$, also size \f$ n\times m\f$
         * @param vardim      the dimension in which the variables are stored
		 */
		template<class V, class M, class L>
		void softmax_derivative(cuv::tensor<V, M,L>& dst, const cuv::tensor<V, M,L>& softmax_act, const cuv::tensor<V,M,L>& residual, unsigned int vardim=1);

        /**
         * @brief Do a gradient update step using AdaGrad.
         * 
         * @param W 	Destination matrix
         * @param dW	The gradient of W. This is a tensor of same shape as W. 
         * @param sW	The sum of the squared gradients for each component as W (therefore also same shape as W).
         * @param learnrate Scalar learnreate 
         * @param delta	added in denominator of adagrad
         * @param decay	(optional) Scalar L2 penalty 
         * @param sparsedecay	(optional) Scalar L1 penalty 
         * 
         */
        template<class V, class M, class L>
            void adagrad(tensor<V,M,L>& W, const tensor<V,M,L>& dW, tensor<V,M,L>& sW, const float& learnrate, const float& delta, const float& decay = 0.0f, const float& sparsedecay=0.0f);

        /**
         * @brief Do a gradient update step using RMSPROP.
         * 
         * @param W 	Destination matrix
         * @param dW	The gradient of W. This is a tensor of same shape as W. 
         * @param sW	The sum of the squared gradients for each component as W (therefore also same shape as W).
         * @param learnrate Scalar learnreate 
         * @param delta	added in denominator of rmsprop
         * @param decay	(optional) Scalar L2 penalty 
         * @param sparsedecay	(optional) Scalar L1 penalty 
         * @param avg_grad time constant to average gradient squares with (0.9 means keep most of old average)
         * 
         */
        template<class V, class M, class L>
            void rmsprop(tensor<V,M,L>& W, const tensor<V,M,L>& dW, tensor<V,M,L>& oldW, const float& learnrate, const float& delta, const float& decay = 0.0f, const float& sparsedecay=0.0f, const float& grad_avg=0.9f);

        /**
         * @brief Do a gradient update step using Nesterov accelerated RMSPROP.
         *
         * @param W 	Destination matrix
         * @param dW	The gradient of W. This is a tensor of same shape as W.
         * @param oldW	The weight-update of the privious learning-step for each component as W (therefore also same shape as W).
         * @param sW	The sum of the squared gradients for each component as W (therefore also same shape as W).
         * @param learnrates The per weight learningrates (therefore also same shape as W).
         * @param learnrate Scalar learnreate
         * @param momentum Scalar momentum-constant
         * @param avg_grad time constant to average gradient squares with (0.9 means keep most of old average)
         * @param step_adapt adaptable step rate constant
         */
        template<class V, class M, class L>
            void na_rmsprop(tensor<V,M,L>& W, const tensor<V,M,L>& dW, tensor<V,M,L>& oldW, tensor<V,M,L>& sW, tensor<V,M,L>& learnrates, const float& learnrate, const float& momentum, const float& grad_avg, const float& step_adapt, const float& delta);
    }
} };

#endif
