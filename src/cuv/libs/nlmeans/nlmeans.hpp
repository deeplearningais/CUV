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

#ifndef __NLMEANS_HPP__
#define __NLMEANS_HPP__

#include <cuv/basics/tensor.hpp>

namespace cuv{
	namespace libs
	{
		/// non-local means
		namespace nlmeans
		{
			/**
			 * @addtogroup libs
			 * @{
			 * @defgroup nlmeans Non-Local means
			 * @{
			 */
			template<class T>
				void filter_nlmean(cuv::tensor<T,dev_memory_space>& dst, const cuv::tensor<T,dev_memory_space>& src, bool threeDim=false);
			template<class T>
				void filter_nlmean(cuv::tensor<T,dev_memory_space,row_major>& dst, const cuv::tensor<T,dev_memory_space,row_major>& src, int search_radius, int filter_radius, float sigma, float dist_sigma=0.f, float step_size=1.f, bool threeDim=false, bool verbose=false);
			/**
			 * @}
			 * @}
			 */
			
		}
	}
}

#endif /* __NLMEANS_HPP__ */

