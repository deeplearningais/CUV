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
 * @file matrix.hpp
 * @brief general base class for 2D matrices
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2010-03-21
 */
#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include <tools/cuv_general.hpp>

namespace cuv{

	template<class V, class T, class I>
	class dia_matrix;
		

/**
 * @brief Basic matrix class
 *
 * This matrix class is the parent of all other matrix classes and has all the basic attributes that all matrices share.
 * This class is never actually instanciated.
 */
template<class __value_type, class __index_type>
class matrix
	{
	  public:
		  typedef __value_type value_type;	///< Type of the entries of matrix
		  typedef __index_type index_type;	///< Type of indices
		  template <class Archive, class V, class I> friend void serialize(Archive&, dia_matrix<V,host_memory_space, I>&, unsigned int) ; ///< serialization function to save matrix
	  protected:
		  index_type m_width; ///< Width of matrix
		  index_type m_height; ///< Heigth of matrix
		public:
		  /** 
		   * @brief Basic constructor: set width and height of matrix but do not allocate any memory
		   * 
		   * @param h Height of matrix
		   * @param w Width of matrix
		   */
		  matrix(const index_type& h, const index_type& w) 
			: m_width(w), m_height(h)
			{
			}
		  virtual ~matrix(){ ///< Empty destructor
		  }
		  /** 
		   * @brief Resizing matrix: changing width and height without changing memory layout
		   * 
		   * @param h New height of matrix 
		   * @param w New width of matrix
		   */
		  inline void resize(const index_type& h, const index_type& w) 
		  {
			  cuvAssert(w*h == m_width*m_height);
			  m_width=w;
			  m_height=h;
		  }
		  inline index_type w()const  { return m_width;                } ///< Return matrix width
		  inline index_type h()const  { return m_height;               } ///< Return matrix height
		  inline index_type n()const  { return w()*h();                } ///< Return number of entries in matrix
		  //virtual void alloc() = 0; 									 ///< Purely virtual
		  //virtual void dealloc() = 0; 									 ///< Purely virtual
	};
}

#endif /* __MATRIX_HPP__ */
