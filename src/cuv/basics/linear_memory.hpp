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
 * @file linear_memory.hpp
 * @brief base class for linear_memory
 * @ingroup basics
 * @author Hannes Schulz, Andreas Mueller
 * @date 2010-03-21
 */
#ifndef __LINEAR_MEMORY_HPP__
#define __LINEAR_MEMORY_HPP__
#include <iostream>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/tools/meta_programming.hpp>
#include <cuv/basics/reference.hpp>

namespace cuv{

/**
 * @brief Basic linear memory class
 *
 * This linear memory class is the generic storage classes and is repsonsible for allocation/deletion.
 */
template<class __value_type, class __memory_space_type, class TPtr=const __value_type*, class __index_type=unsigned int>
class linear_memory{
	public:
	  typedef __value_type          value_type;	 ///< Type of the entries of memory
	  typedef const value_type               const_value_type;	///< Type of the entries of matrix
	  typedef __index_type index_type;	 ///< Type of indices
	  typedef __memory_space_type memory_space_type; ///< Indicates whether this is a host or device vector
	  typedef TPtr                pointer_type; ///< Type of stored pointer, should be const or not-const value_type*
	  template <class Archive, class V, class I> friend 
		  void serialize(Archive&, linear_memory<V,memory_space_type,I>&, unsigned int) ; ///< serialize/deserialize the linear_memory to/from an archive
	  typedef linear_memory<value_type, memory_space_type, TPtr, index_type> my_type; ///< Type of this linear_memory
	  typedef reference<value_type, memory_space_type,index_type> reference_type; ///< Type of references returned by operators
	  typedef const reference_type const_reference_type; ///< Type of references returned by operators
	protected:
	  pointer_type m_ptr; ///< Pointer to actual entries in memory
	  bool         m_is_view; ///< Indicates whether this linear_memory owns the memory or is just a view
	  index_type   m_size; ///< Length of linear_memory
	  allocator<value_type, index_type, memory_space_type> m_allocator;
	public:
	  /*
	   * Member Access
	   */

	  /** 
	   * @brief Return pointer to matrix entries
	   */
	  inline const pointer_type ptr()const{ return m_ptr;  }	
	  /** 
	   * @brief Return pointer to matrix entries
	   */
	  inline       pointer_type ptr()     { return m_ptr;  }
	  /** 
	   * @brief Return true if this linear_memory is a view and doesn't own the memory
	   */
	  inline bool is_view() const         { return m_is_view; }
	  /**
	   * @brief Return length of linear_memory
	   */
	  inline index_type size() const         { return m_size; }
	  /**
	   * @brief set length of linear_memory. If this is longer than the
	   * original size, this will destroy the content due to reallocation!
	   */
	  void set_size(const index_type& s) {
		  if(m_size < s){
			  dealloc();
			  m_size = s;
			  alloc();
			  return;
		  }
		  m_size = s;
	  }
	  /** 
	   * @brief Return size of linear_memory in memory
	   */
	  inline size_t memsize()       const{ return size() * sizeof(value_type); } 
	  /*
	   * Construction
	   */
	  /** 
	   * @brief Empty constructor. Creates empty linear_memory (allocates no memory)
	   */
	  linear_memory():m_ptr(NULL),m_is_view(false),m_size(0) {} 
	  /** 
	   * @brief Creates linear_memory of lenght s and allocates memory
	   * 
	   * @param s Length of linear_memory
	   */
	  linear_memory(index_type s):m_ptr(NULL),m_is_view(false),m_size(s) {
		  alloc();
	  }
	  /** 
	   * @brief Copy-Constructor
	   */
	  linear_memory(const my_type& o):m_ptr(NULL),m_is_view(false),m_size(o.size()) {
		  alloc();
		  m_allocator.copy(m_ptr,o.ptr(),size(),memory_space_type());
	  }
	  /** 
	   * @brief Copy-Constructor for other memory spaces
	   */
	  template<class OM, class OP>
	  linear_memory(const linear_memory<__value_type, OM, OP,__index_type>& o):m_ptr(NULL),m_is_view(false),m_size(o.size()) {
		  alloc();
		  m_allocator.copy(m_ptr,o.ptr(),size(),OM());
	  }
	  /** 
	   * @brief Creates linear_memory from pointer to entries.
	   * 
	   * @param s Length of linear_memory
	   * @param p Pointer to entries 
	   * @param is_view If true will not take responsibility of memory at p. Otherwise will dealloc p on destruction.
	   */
	  linear_memory(index_type s,pointer_type p, bool is_view):m_ptr(p),m_is_view(is_view),m_size(s) {
		  alloc();
	  }
	  /** 
	   * @brief Deallocate memory if is_view is false.
	   */
	  virtual ~linear_memory(){
		  dealloc();
	  } 

	  /*
	   * Memory Management
	   */

	  /** 
	   * @brief Allocate memory
	   */
	  void alloc(){
		  if (! m_is_view) {
			  cuvAssert(m_ptr == NULL)
			  m_allocator.alloc( &m_ptr,m_size);
		  }
	  } 

	  /** 
	   * @brief Deallocate memory if not a view
	   */
	  void dealloc(){
		  if (m_ptr && ! m_is_view)
			m_allocator.dealloc(&m_ptr);
		  m_ptr=NULL;
	  }


		/** 
		 * @brief Copy linear_memory.
		 * 
		 * @param o Source linear_memory
		 * 
		 * @return copy to *this
		 *
		 */
	  my_type& 
		  operator=(const my_type& o){
			 
			if(this->size() != o.size()){
			  this->dealloc();
			  m_size = o.size();
			  this->alloc();
			}
			m_allocator.copy(m_ptr,o.ptr(),size(),memory_space_type());
			  
			return *this;
		  }

		/** 
		 * @brief Copy linear_memory from other memory type.
		 * 
		 * @param o Source linear_memory
		 * 
		 * @return copy to *this
		 *
		 */
	  template<class OM, class OP>
	  my_type& 
	  
		  operator=(const linear_memory<value_type, OM, OP,index_type>& o){
			if(this->size() != o.size()){
			  this->dealloc();
			  m_size = o.size();
			  this->alloc();
			}
			m_allocator.copy(m_ptr,o.ptr(),size(),OM());
			return *this;
		  }

		reference_type
		operator[](const index_type& idx)     ///< Return entry at position t
			{
				return reference_type(m_ptr+idx);
			}
		const_reference_type
		operator[](const index_type& idx)const///< Return entry at position t
			{
				return const_reference_type(m_ptr+idx);
			}

		/** 
		 * @brief Set entry idx to val
		 * 
		 * @param idx Index of which entry to change 
		 * @param val New value of entry
		 */
		void set(const index_type& idx, const value_type& val){
			(*this)[idx] = val;
		}

}; // linear_memory


}; // cuv

#endif

