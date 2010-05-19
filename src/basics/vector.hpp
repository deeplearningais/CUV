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
 * @file vector.hpp
 * @brief base class for vectors
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2010-03-21
 */
#ifndef __VECTOR_HPP__
#define __VECTOR_HPP__
#include <tools/cuv_general.hpp>
#include <iostream>

namespace cuv{

template <class value_type, class index_type>
void alloc( value_type** ptr, index_type memsize, dev_memory_space);
template <class value_type>
void dealloc( value_type** ptr, dev_memory_space);
template <class value_type, class index_type>
void entry_set(value_type* ptr, index_type idx, value_type val, dev_memory_space);
template <class value_type, class index_type>
value_type entry_get(const value_type* ptr, index_type idx, dev_memory_space);

template <class value_type, class index_type>
void alloc( value_type** ptr, index_type memsize, host_memory_space);
template <class value_type>
void dealloc( value_type** ptr, host_memory_space);
template <class value_type, class index_type>
void entry_set(value_type* ptr, index_type idx, value_type val, host_memory_space);
template <class value_type, class index_type>
value_type entry_get(const value_type* ptr, index_type idx, host_memory_space);



/**
 * @brief Basic vector class
 *
 * This vector class is the parent of all other vector classes and has all the basic attributes that all matrices share.
 * This class is never actually instanciated.
 */
template<class __value_type, class __memory_space_type, class __index_type=unsigned int>
class vector{
	public:
	  typedef __value_type value_type;	 ///< Type of the entries of matrix
	  typedef __index_type index_type;	 ///< Type of indices
	  typedef __memory_space_type memory_space_type; ///< Indicates whether this is a host or device vector
	  template <class Archive, class V, class I> friend void serialize(Archive&, vector<V,memory_space_type,I>&, unsigned int) ; ///< serialize/deserialize the vector to/from an archive
	  typedef vector<value_type, memory_space_type, index_type> my_type; ///< Type of this vector
	protected:
	  value_type* m_ptr; ///< Pointer to actual entries in memory
	  bool        m_is_view; ///< Indicates whether this vector owns the memory or is just a view
	  index_type  m_size; ///< Length of vector
			
	public:
	  /*
	   * Member Access
	   */

	  /** 
	   * @brief Return pointer to matrix entries
	   */
	  inline const value_type* ptr()const{ return m_ptr;  }	
	  /** 
	   * @brief Return pointer to matrix entries
	   */
	  inline       value_type* ptr()     { return m_ptr;  }
	  /** 
	   * @brief Return length of vector
	   */
	  inline index_type size() const         { return m_size; }
	  /** 
	   * @brief Return size of vector in memory
	   */
	  inline size_t memsize()       const{ return size() * sizeof(value_type); } 
	  /*
	   * Construction
	   */
	  /** 
	   * @brief Empty constructor. Creates empty vector (allocates no memory)
	   */
	  vector():m_ptr(NULL),m_is_view(false),m_size(0) {} 
	  /** 
	   * @brief Creates vector of lenght s and allocates memory
	   * 
	   * @param s Length of vector
	   */
	  vector(index_type s):m_ptr(NULL),m_is_view(false),m_size(s) {
		  alloc();
	  }
	  /** 
	   * @brief Creates vector from pointer to entries.
	   * 
	   * @param s Length of vector
	   * @param p Pointer to entries 
	   * @param is_view If true will not take responsibility of memory at p. Otherwise will dealloc p on destruction.
	   */
	  vector(index_type s,value_type* p, bool is_view):m_ptr(p),m_is_view(is_view),m_size(s) {
		  alloc();
	  }
	  /** 
	   * @brief Deallocate memory if is_view is false.
	   */
	  virtual ~vector(){
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
			  cuv::alloc<value_type, index_type>( &m_ptr,m_size,memory_space_type());	
		  }
	  } 

	  /** 
	   * @brief Deallocate memory if not a view
	   */
	  void dealloc(){
		  if (m_ptr && ! m_is_view)
			cuv::dealloc<value_type>(&m_ptr,memory_space_type());	
		  m_ptr=NULL;
	  }


		/** 
		 * @brief Assignment operator. Assigns memory belonging to source to destination and sets source memory pointer to NULL (if source is not a view)
		 * 
		 * @param o Source matrix
		 * 
		 * @return Matrix of same size and type of o that now owns vector of entries of o.
		 *
		 * If source vector is a view, the returned vector is a view, too.
		 */
	  my_type& 
		  operator=(const my_type& o){
			  if(this==&o) return *this;
			  this->dealloc();
			  this->m_ptr = o.m_ptr;
			  this->m_is_view = o.m_is_view;
			  this->m_size = o.m_size;
			  if(!m_is_view){
				  // transfer ownership of memory (!)
				  (const_cast< my_type *>(&o))->m_ptr = NULL;
			  }
			  return *this;
		  }

		value_type operator[](const index_type& idx)const///< Return entry at position t
			{
				return entry_get<value_type,index_type>(m_ptr,idx,memory_space_type());
			}

		/** 
		 * @brief Set entry idx to val
		 * 
		 * @param idx Index of which entry to change 
		 * @param val New value of entry
		 */
		void set(const index_type& idx, const value_type& val){
				entry_set<value_type,index_type>(m_ptr,idx,val,memory_space_type());
		}

}; // vector

/** 
 * @brief Allocate memory for host matrices 
 * 
 * @param ptr Address of pointer which will be set to allocated memory
 * @param size Size of array which should be allocated
 * 
 * This is the instance of the alloc function that is called by host vectors.
 */
template<class value_type, class index_type>
void alloc( value_type** ptr, index_type size, host_memory_space) {
	*ptr = new value_type[size];
}

/** 
 * @brief Deallocate memory for host matrices
 * 
 * @param ptr Address of pointer that will be freed
 * 
 * This is the instance of the dealloc function that is called by host vectors.
 */
template<class value_type>
void dealloc( value_type** ptr, host_memory_space) {
	delete[] *ptr;
	*ptr = NULL;
}

/** 
 * @brief Setting entry of host vector at ptr at index idx to value val
 * 
 * @param ptr Address of array in memory
 * @param idx Index of value to set
 * @param val Value to set vector entry to
 * 
 */
template <class value_type, class index_type>
void entry_set(value_type* ptr, index_type idx, value_type val, host_memory_space) {
	ptr[idx]=val;
}


/** 
 * @brief Getting entry of host vector at ptr at index idx
 * 
 * @param ptr Address of array in memory
 * @param idx Index of value to get
 * 
 * @return 
 */
template <class value_type, class index_type>
value_type entry_get(const value_type* ptr, index_type idx, host_memory_space) {
	return ptr[idx];
}

}; // cuv

#endif
