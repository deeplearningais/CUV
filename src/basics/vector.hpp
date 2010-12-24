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
 * @author Hannes Schulz, Andreas Mueller
 * @date 2010-03-21
 */
#ifndef __VECTOR_HPP__
#define __VECTOR_HPP__
#include <tools/cuv_general.hpp>
#include <iostream>
#include <vector_ops/vector_ops.hpp>

namespace cuv{

//host and device	
template <class value_type, class index_type, class memory_space_type>
struct allocator{
	void alloc( value_type** ptr, index_type memsize)const;
	void dealloc( value_type** ptr)const;
};

//host
template <class value_type, class index_type>
void entry_set(value_type* ptr, index_type idx, value_type val, host_memory_space);
template <class value_type, class index_type>
value_type entry_get(const value_type* ptr, index_type idx, host_memory_space);

//device
template <class value_type, class index_type>
void entry_set(value_type* ptr, index_type idx, value_type val, dev_memory_space);
template <class value_type, class index_type>
value_type entry_get(const value_type* ptr, index_type idx, dev_memory_space);

template<class __vector_type, class __value_type>
void fill(__vector_type& v, const __value_type& p);

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
	  allocator<value_type, index_type, memory_space_type> m_allocator;
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
	   * @brief Return true if this vector is a view and doesn't own the memory
	   */
	  inline bool is_view() const         { return m_is_view; }
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
	   * @brief Copy-Constructor
	   */
	  vector(const my_type& o):m_ptr(NULL),m_is_view(false),m_size(o.size()) {
		  alloc();
		  copy(*this,o);
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
		 * @brief Assign a scalar to a vector.
		 * 
		 * @param scalar   the scalar to assign to all positions in *this
		 * 
		 * @return Reference to *this
		 *
		 */
	  my_type& 
		  operator=(const value_type& scalar){
			 
			fill(*this, scalar);
			return *this;
		  }
		/** 
		 * @brief Copy vector.
		 * 
		 * @param o Source vector
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
			copy(*this, o);
			  
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

template<class value_type, class index_type>
struct allocator<value_type,index_type,host_memory_space>{
	/** 
	 * @brief Allocate memory for host matrices 
	 * 
	 * @param ptr Address of pointer which will be set to allocated memory
	 * @param size Size of array which should be allocated
	 * 
	 * This is the instance of the alloc function that is called by host vectors.
	 */
	void alloc( value_type** ptr, index_type size) const{
		*ptr = new value_type[size];
	}
	/** 
	 * @brief Deallocate memory for host matrices
	 * 
	 * @param ptr Address of pointer that will be freed
	 * 
	 * This is the instance of the dealloc function that is called by host vectors.
	 */
	void dealloc( value_type** ptr)const {
		delete[] *ptr;
		*ptr = NULL;
	}
};


template<class value_type, class index_type>
struct allocator<const value_type,index_type,host_memory_space>{
	/** 
	 * @brief Allocate memory for host matrices - const allocator should never be called!
	 * 
	 * @param ptr Address of pointer which will be set to allocated memory
	 * @param size Size of array which should be allocated
	 * 
	 * This is the instance of the alloc function that is called by const host vectors.
	 */
	void alloc(const value_type** ptr, index_type size) const{
		cuvAssert(false);
	}
	/** 
	 * @brief Deallocate memory for host matrices- const allocator should never be called!
	 * 
	 * @param ptr Address of pointer that will be freed
	 * 
	 * This is the instance of the dealloc function that is called by host vectors.
	 */
	void dealloc(const value_type** ptr)const {
		cuvAssert(false);
	}
};


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
