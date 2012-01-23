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

/** 
 * @file image.hpp
 * @brief general base class for images 
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2011-05-19
 */
#ifndef __IMAGE_HPP__
#define __IMAGE_HPP__
#include <cuv/basics/tensor.hpp>

namespace cuv
{
	/**
	 * a wrapper around a tensor to provide an interleaved (e.g. RGBRGBRGB...) image
	 *
	 * the internal tensor is a _strided_ tensor.
	 *
	 */
	template<int NumChannels, class __value_type, class __memory_space_type>
	class 
	interleaved_image{
		public:
			/// the type of the wrapped tensor: Row-major and strided!
			typedef cuv::tensor<__value_type,__memory_space_type,row_major> tensor_type;
			/// the index type
			typedef typename tensor_type::index_type index_type;
			/// the type of returned references
			typedef typename tensor_type::reference_type reference_type;
		private:
			tensor_type m_tens;
			unsigned int m_height;
			unsigned int m_width;
		public:
			static const int num_channels = NumChannels;
			/**
			 * construct an interleaved image based on dimensions
			 *
			 * @param h height
			 * @param w width
			 * @param c number of channels
			 */
			interleaved_image(unsigned int h, unsigned int w, unsigned int c=1)
			:	m_tens(extents[h][w*NumChannels]),
				m_height(h),
				m_width(w)
			{
			}
			/**
			 * copy-construct an interleaved_image
			 * @param o source image
			 */
			interleaved_image(const interleaved_image& o)
			:	m_tens(o.tens()),
				m_height(o.height()),
				m_width(o.width())
			{
			}
			
			/// @return the width of the image
			inline index_type width()const{ return m_width; }
			/// @return the height of the image
			inline index_type height()const{ return m_height; }
			/// @return the number of channels
			inline index_type channels()const{ return NumChannels; }

			/// @return the wrapped tensor
			inline const tensor_type& tens()const{ return m_tens; } 

			/**
			 * element access
			 *
			 * @param i index along height
			 * @param j index along width
			 * @param c index of channel
			 */
			reference_type
			operator()(index_type i, index_type j, index_type c=0){
				return m_tens(i,j*NumChannels+c);
			}
	
			/**
			 * const element access
			 *
			 * @param i index along height
			 * @param j index along width
			 * @param c index of channel
			 */
			const reference_type
			operator()(index_type i, index_type j, index_type c=0)const{
				return m_tens(i,j*NumChannels+c);
			}

			/**
			 * assignment operator
			 *
			 * @param o source image
			 */
			interleaved_image&
			operator=(const interleaved_image& o){
				m_width    = o.width();
				m_height   = o.height();
				m_tens     = o.tens();
				return *this;
			}
	};
}


#endif /* __IMAGE_HPP__ */
