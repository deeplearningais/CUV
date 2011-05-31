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
 * @file cuda_array.hpp
 * @brief wrapper around cuda_array
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2010-10-22
 *
 */
#ifndef __CUDA_ARRAY_HPP__
#define __CUDA_ARRAY_HPP__

#include <cuv/basics/tensor.hpp>
#include <cuv/basics/matrix.hpp>

class cudaArray; // forward declaration of cudaArray so we do not need to include cuda headers here


namespace cuv
{
	/** 
	 * @brief Wrapper for a 2D CUDAArray
	 */
	template<class __value_type, class __memory_space_type, class __index_type = unsigned int >
	class cuda_array 
	:        public matrix<__value_type, __index_type>{
		public:
		  typedef __memory_space_type								  memory_space_type;///< Indicates whether matrix resides on host or device
		  typedef matrix<__value_type, __index_type>				  base_type;		///< Basic matrix type
		  typedef typename base_type::value_type 					  value_type; 		///< Type of matrix entries
		  typedef typename base_type::index_type 					  index_type;		///< Type of indices
		  typedef cuda_array<value_type,memory_space_type,index_type>  my_type;	        ///< Type of this object
		  using base_type::m_width;
		  using base_type::m_height;
		  index_type m_depth;

		private:
		  cudaArray*         m_ptr;   ///< data storage in cudaArray
		  unsigned int       m_dim;   ///< the dimension of a single value (can be 1 or 4)

		public:
		  /**
		   * @brief Construct uninitialized memory with given width/height.
		   * @param dim dimension of a single point in the array (can be 1 or 4)
		   *
		   * when type is float and dim=1 then the cudaArray is float
		   *
		   * when type is float and dim=4 then the cudaArray is float4
		   */
		  cuda_array(const index_type& height, const index_type& width, const index_type& depth=1, const unsigned int dim=1)
			  :base_type(height, width)
			  ,m_depth(depth)
			  ,m_ptr(NULL)
			  ,m_dim(dim)
		  {
			  alloc();
		  }

		  /**
		   * @brief Construct by copying.
		   * @param dim dimension of a single point in the array (can be 1 or 4)
		   *
		   * when type is float and dim=1 then the cudaArray is float
		   *
		   * when type is float and dim=4 then the cudaArray is float4
		   */
		  template<class S>
		  cuda_array(const tensor<value_type,S,row_major>& src, const unsigned int dim=1) ///< construct by copying a dense matrix
		  :base_type(0, 0)
		  , m_ptr(NULL)
		  , m_dim(dim)
		  {
			  if(src.ndim()==2){
				  m_height = src.shape()[0];
				  m_width  = src.shape()[1];
				  m_depth  = 1;
			  }else if(src.ndim()==3){
				  m_depth  = src.shape()[0];
				  m_height = src.shape()[1];
				  m_width  = src.shape()[2];
			  }
			  alloc();
			  assign(src);
		  }
		  ~cuda_array(){ ///< when destroying, delete associated memory
			  dealloc();
		  }
		  inline index_type w()const{return m_width;}           ///< width 
		  inline index_type h()const{return m_height;}          ///< height 
		  inline index_type d()const{return m_depth;}          ///< depth
		  inline index_type n()const{return m_width*m_height*m_depth;}  ///< number of elements
		  inline index_type dim()const{return m_dim; }          ///< size of a single array element (in units of sizeof(value_type))
		  inline       cudaArray* ptr()      {return m_ptr;}    ///< the wrapped cudaArray
		  inline const cudaArray* ptr() const{return m_ptr;}    ///< the wrapped cudaArray
		  void alloc();                                         ///< allocate memory
		  void dealloc();                                       ///< free memory
		  /**
		   * @brief assign memory
		   *
		   * src.w() should be the same as this->w()*this->dim
		   *
		   * src.h() should be the same as this->h()
		   */
		  void assign(const tensor<__value_type,dev_memory_space,row_major>& src);  
		  void assign(const tensor<__value_type,dev_memory_space,row_major,memory2d_tag>& src);  
		  /**
		   * @brief assign memory
		   *
		   * src.w() should be the same as this->w()*this->dim
		   *
		   * src.h() should be the same as this->h()
		   */
		  void assign(const tensor<__value_type,host_memory_space,row_major>& src);

		  /**
		   * broken/useless.
		   */
		  __value_type operator()(const __index_type& i, const __index_type& j)const;
	};
}

#endif /* __CUDA_ARRAY_HPP__ */
