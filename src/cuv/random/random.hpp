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


#include <cuv/basics/tensor.hpp>


#ifndef __RANDOM_HPP__
#define __RANDOM_HPP__

namespace cuv{

 /** @addtogroup random 
  * @{
  */

	/** 
	 * @brief Fill a matrix/vector with random numbers uniformly distributed between zero and one
	 * 
	 * @param dst Destionation matrix/vector
	 */
	template<class V, class M, class T> void fill_rnd_uniform(tensor<V, M, T>& dst);  

	/** 
	 * @brief Binarize a matrix/vector to 1 or 0 with probability given by current values
	 * 
	 * @param dst	Destionation matrix/vector 
	 */
	template<class V, class M, class T> void rnd_binarize(tensor<V, M, T>& dst);      

	/** 
	 * @brief Add random numbers (normally distributed, mean 0) to a matrix/vector
	 * 
	 * @param dst Destination matrix/vector
	 * @param std Standard deviation of normal distribution used
	 */
	template<class V, class M, class T> void add_rnd_normal(tensor<V, M, T>& dst,const float& std=1.0f);  

	/** 
	 * @brief Initialize Mersenne twister to generate random numbers on GPU
	 * 
	 * @param seed Seed for initialization
	 *
	 * This function has to be called exactly _once_ before making use of any random functions.
	 */
	void initialize_mersenne_twister_seeds(unsigned int seed = 0); 

	/** 
	 * @brief destruction counterpart to @see initialize_mersenne_twister_seeds
	 * 
	 */
	void deinit_rng(unsigned int seed = 0); 

 /** @} */ // end of group random

} // cuv

#endif
