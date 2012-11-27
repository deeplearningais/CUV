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

#ifndef __CUV_INTIMG_HPP__
#define __CUV_INTIMG_HPP__
#include<cuv/basics/tensor.hpp>

namespace cuv{
	/// integral image computation
	namespace integral_img
	{
		/**
		 * @addtogroup libs
		 * @{
		 * @defgroup integral_img Integral Image
		 * @{
		 */

		/**
		 * calculate the integral image
		 *
		 * this applies \see scan twice, transposing in between.
		 *
		 * @param src source
		 * @param dst destination
		 */
		template<class V1, class V2, class T, class M>
		void integral_image(cuv::tensor<V1, T, M>& dst, const cuv::tensor<V2, T, M>& src);

		/**
		 * integrate rows of an image
		 * @param src source
		 * @param dst destination
		 */
		template<class V1, class V2, class T, class M>
		void scan(cuv::tensor<V1, T, M>& dst, const cuv::tensor<V2, T, M>& src);

        /**
         * calculates many integral images in parallel, for data given in
         * format required by Alex' convolutions. 
         *
         * The input (and output) is assumed to be row-major and
         *
         *   nChannels x nRows x nCols x nImages.
         *
         * every channel of every image is integrated separately.
         *
         * We compute the /exclusive/ scan, s.t. the output is
         *
         *   nChannels x (nRows+1) x (nCols+1) x nImages.
         *
         * @param src source
         * @param dst destination
         */
        template<class V, class M>
        void integral_image_4d(cuv::tensor<V,M>& dst, const cuv::tensor<V,M>& src);
		/**
		 * @}
		 * @}
		 */
	}
};

#endif
