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





#ifndef MOVE_HPP_
#define MOVE_HPP_
#include<cuv/basics/tensor.hpp>

namespace cuv
{

	/** 
	 * @defgroup imageops Operations on Images
	 * @brief Write a moved version of each image (a column in src) to dst.
	 *
	 * Assumptions: 
	 * - (n*num_maps by m) matrix, 
	 *   where n=image_width*image_height is an image
	 * - images are in RGBA interleaved format(num_maps=4), A channel is ignored.
	 * - images can also be in grayscale (num_maps=1).
	 *
	 * @todo previously non-existent pixels at the border are filled... how?
	 * 
	 * @}
	 */
	 
	 /** 
	 * @brief Shift images by given amount
	 * 
	 * @param dst where the moved images are written
	 * @param src unsigned char where original images are taken from
	 * @param src_image_size  width and height of image in source
	 * @param dst_image_size  width and height of image in destination
	 * @param src_num_maps  how many maps there are in src
	 * @param xshift how much to shift right
	 * @param yshift how much to shift down
	 */
	template<class __value_typeA, class __value_typeB, class __memory_space_type, class __memory_layout_type>
	void image_move(tensor<__value_typeA,__memory_space_type,__memory_layout_type>& dst, const tensor<__value_typeB,__memory_space_type,__memory_layout_type>& src, const unsigned int& image_width, const unsigned int& image_height, const unsigned int& num_maps, const int& xshift, const int& yshift);
};



#endif /* MOVE_HPP_ */

