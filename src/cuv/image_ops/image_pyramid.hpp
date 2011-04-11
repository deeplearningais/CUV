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
 * @file image_pyramid.hpp
 * @brief classes/methods dealing with the construction of image pyramids
 * @ingroup image_ops
 * @author Hannes Schulz
 * @date 2010-10-22
 */

#ifndef __IMAGE_PYRAMID_HPP__
#define __IMAGE_PYRAMID_HPP__

#include <boost/ptr_container/ptr_vector.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/basics/cuda_array.hpp>

namespace cuv{

	/**
	 * @brief image pyramid decreasing in size logarithmically.
	 */
	template <class __matrix_type>
	class image_pyramid
	{
	public:
		typedef __matrix_type              matrix_type;            ///< the type of the contained matrices
		typedef typename matrix_type::value_type    value_type;    ///< the type of one value in the matrix
		typedef typename matrix_type::memory_layout memory_layout; ///< the memory layout (assumed to be row_major!)
		typedef typename matrix_type::index_type    index_type;    ///< the index_type

		/**
		 * construct an image pyramid.
		 *
		 * @param img_h height of one image at base
		 * @param img_w width of one image at base
		 * @param depth depth of the pyramid (0..depth-1)
		 * @param dim   the pixel dimension (can be 1 or 3)
		 */
		image_pyramid( int img_h, int img_w, int depth, int dim );
		/**
		 * Get a view on a channel in the pyramid.
		 * @param depth    level of the pyramid
		 * @param channel  channel (red, for example)
		 */
		matrix_type* get(int depth, int channel);
		matrix_type* get_all_channels(int depth);
		inline int dim(){return m_dim;}                           ///< dimension of a single value in the matrix
		inline unsigned int base_h(){return m_base_height;}       ///< image height at base of matrix
		inline unsigned int base_w(){return m_base_width;}        ///< image width at base of matrix
		inline unsigned int depth(){return m_matrices.size();}    ///< depth of pyramid

		/**
		 * @brief build the pyramid given an image.
		 *
		 * @param src                  pyramid represents scaled versions of this
		 * @param interleaved_channels can be either 1 (grayscale) or 4 (RGBA, only RGB is used though).
		 */
		template<class __arg_matrix_type>
		void build(const __arg_matrix_type& src, const unsigned int interleaved_channels){
			typedef typename __arg_matrix_type::value_type   argval_type;
			typedef typename __arg_matrix_type::memory_space_type argmemspace_type;
			typedef typename __arg_matrix_type::index_type   argindex_type;
			typedef cuda_array<argval_type,argmemspace_type,argindex_type> argca_type;

			/////////////
			// create base image at level 0
			////////////
			if(    src.h() == m_base_height*m_dim // the image dimensions match the input --> just copy.
				&& src.w() == m_base_width
			){
				//std::cout << "Copycase"<<std::endl;
					m_matrices[0]=src;
			}
			else if(   interleaved_channels == 4
					&& m_dim                == 3
			){
				//std::cout << "Colorcase"<<std::endl;
					argca_type cpy(src,4);
					matrix_type& dst = m_matrices[0];
					gaussian_pyramid_downsample(dst, cpy, interleaved_channels);
			}
			else if(src.h() > m_base_height*m_dim  // the image dimensions are too large: downsample to 1st level of pyramid
				&&  src.w() > m_base_width
			){
				//std::cout << "Multichannel case"<<std::endl;
				for(int i=0;i<m_dim;i++){
					const __arg_matrix_type view(src.h()/m_dim, src.w(),(argval_type*)src.ptr(),true);
					argca_type cpy(view);
					matrix_type* dstview = get(0,i);
					gaussian_pyramid_downsample(*dstview, cpy,1);
					delete dstview;
				}
			}else{
				cuvAssert(false);
			}

			/////////////
			// fill upper levels
			////////////
			for(int i=1;i<m_matrices.size();i++){
				for(int d=0;d<m_dim;d++){
					matrix_type* srcview = get(i-1,d);
					argca_type cpy(*srcview);
					matrix_type* dstview = get(i,  d);
					gaussian_pyramid_downsample(*dstview, cpy,1);
					delete dstview;
					delete srcview;
				}
			}

		}
	private:
		boost::ptr_vector<matrix_type> m_matrices;
		unsigned int m_dim;
		unsigned int m_base_width;
		unsigned int m_base_height;
	};
	
	template <class __matrix_type>
	typename image_pyramid<__matrix_type>::matrix_type*
	image_pyramid<__matrix_type>::get(int depth, int channel){
		cuvAssert(depth   < m_matrices.size());
		cuvAssert(channel < m_dim);
		matrix_type& mat = m_matrices[depth];
		//std::cout << "asking for channel "<<channel<<" in matrix of size "<<mat.h()<<"x"<<mat.w()<<std::endl;
		unsigned int w = mat.w();
		unsigned int h = mat.h();
		return new matrix_type(h/m_dim,w,mat.ptr()+channel*w*h/m_dim,true);
	}

	template <class __matrix_type>
	typename image_pyramid<__matrix_type>::matrix_type*
	image_pyramid<__matrix_type>::get_all_channels(int depth){
		cuvAssert(depth   < m_matrices.size());
		matrix_type& mat = m_matrices[depth];
		return &mat;
	}

	template <class __matrix_type>
	image_pyramid<__matrix_type>::image_pyramid( int img_h, int img_w, int depth, int dim )
	:m_base_height(img_h)
	,m_base_width(img_w)
	,m_dim(dim)
	{
		m_matrices.clear();
		for(unsigned int i=0; i<depth;i++){
			//std::cout << "Creating Pyramid Level: "<< img_h<<"*"<<m_dim<<"x"<<img_w<<std::endl;
			m_matrices.push_back(new matrix_type(img_h*m_dim, img_w));
			img_h=ceil(img_h/2.f);
			img_w=ceil(img_w/2.f);
		}
	}

/**
 * @brief sample down an image by a factor of 2
 *
 * @param dst     target matrix; when interleaved_channels is 4,
 *                this should be a matrix which is 3 times as high as src
 * @param src     source matrix; when interleaved_channels is 4, this should have dim=4 set
 * @param interleaved_channels can be 1 (grayscale) or 4 (RGBA)
 */
template<class T,class S, class I>
void gaussian_pyramid_downsample(
	tensor<T,S,row_major>& dst,
	const cuda_array<T,S,I>& src,
	const unsigned int interleaved_channels
);
template<class T,class S, class I>
void gaussian_pyramid_upsample(
	tensor<T,S,row_major>& dst,
	const cuda_array<T,S,I>& src
);

template<class TDest, class T,class S, class I>
void get_pixel_classes(
	tensor<TDest,S,row_major>& dst,
	const cuda_array<T,S,I>&           src,
	float scale_fact
);

template<class T,class S, class I>
void gaussian(
	tensor<T,S,row_major>& dst,
	const cuda_array<T,S,I>& src
);

}
#endif /* __IMAGE_PYRAMID_HPP__ */
