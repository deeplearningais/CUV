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
 * @file memory2d.hpp
 * @brief base class for memory2d
 * @ingroup basics
 */
#ifndef __MEMORY2D_HPP__
#define __MEMORY2D_HPP__
#include <iostream>
#include <vector>
#include <numeric>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/tools/meta_programming.hpp>
#include <cuv/basics/reference.hpp>
#include <cuv/basics/memory.hpp>

namespace cuv{


template<class __value_type, class __memory_space_type, class TPtr, class __index_type>
class linear_memory;

/**
 * @brief Basic 2D memory class
 *
 * This memory2d class is the generic storage classes and is repsonsible for allocation/deletion.
 */
template<class __value_type, class __memory_space_type, class TPtr=const __value_type*, class __index_type=unsigned int>
class memory2d
: public memory<__value_type,__memory_space_type,TPtr,__index_type>
{
	public:
	  typedef memory<__value_type,__memory_space_type,TPtr,__index_type> super_type;
	  typedef typename super_type::value_type value_type; ///< Type of the entries of memory
	  typedef typename super_type::const_value_type const_value_type; ///< Type of the entries of memory
	  typedef typename super_type::index_type index_type; ///< Type indices/dimensions
	  typedef typename super_type::memory_space_type memory_space_type; ///< Indicates whether this is a host or device vector
	  typedef typename super_type::pointer_type      pointer_type; ///< Type of stored pointer, should be const or not-const value_type*
	  typedef typename super_type::reference_type      reference_type; ///< Type of references returned by operator[]
	  typedef typename super_type::const_reference_type      const_reference_type; ///< Type of references returned by operator[]
	  template <class Archive, class V, class I> friend 
		  void serialize(Archive&, memory2d<V,memory_space_type,I>&, unsigned int) ; ///< serialize/deserialize the memory2d to/from an archive
	protected:
	  using super_type::m_ptr;
	  using super_type::m_is_view;
	  using super_type::m_allocator;

	  index_type m_width;
	  index_type m_pitch;
	  index_type m_height;
	public:
	  /*
	   * Member Access
	   */
	  index_type width()const{return m_width;}
	  index_type height()const{return m_height;}
	  index_type pitch()const{return m_pitch;}

	  /**
	   * Set width and height of memory2d using a shape
	   *
	   * For 3D-shapes, we assume that the innermost dimension is a
	   * pixel-type, that is, the actual width is (shape0*shape1, shape2).
	   */ 
	  void set_width_and_height(const std::vector<index_type>&shape){
		  cuvAssert(shape.size()==2 || shape.size()==3);
		  if(shape.size()==3){
			  m_width   = shape[0] * shape[1];
			  m_height  = shape[2];
		  }
		  if(shape.size()==2){
			  m_width   = shape[0];
			  m_height  = shape[1];
		  }
	  }


	  /**
	   * @brief set length of memory2d. 
	   *
	   * @param pitch   OUT the stride in bytes for each dimension
	   * @param shape      IN the shape  in value_types
	   *
	   */
	  void set_size(index_type& pitch, const std::vector<index_type>& shape) {
		  dealloc();
		  set_width_and_height(shape);
		  alloc();
	  }
	  size_t memsize()const{
		  return m_pitch * m_height;
	  }
	  /*
	   * Construction
	   */
	  /** 
	   * @brief Empty constructor. Creates empty memory2d (allocates no memory)
	   */
	  memory2d():m_width(0),m_height(0){} 
	  /** 
	   * @brief Creates memory2d of lenght s and allocates memory
	   * 
	   * @param s Length of memory2d
	   */
	  memory2d(index_type h, index_type w) : m_width(w), m_height(h){
		  alloc();
	  }
	  /** 
	   * @brief Copy-Constructor
	   */
	  memory2d(const memory2d& o):super_type(o.size()) {
		  alloc();
		  m_allocator.copy(m_ptr,o.ptr(),memsize(),memory_space_type());
	  }
	  /** 
	   * @brief Copy-Constructor for other memory spaces
	   */
	  template<class OM, class OP>
	  memory2d(const memory2d<OM, OP,__index_type>& o):super_type(o.size()) {
		  alloc();
		  m_allocator.copy(m_ptr,o.ptr(),memsize(),OM());
	  }
	  /** 
	   * @brief Creates memory2d from pointer to entries.
	   * 
	   * @param s Length of memory2d
	   * @param p Pointer to entries 
	   * @param is_view If true will not take responsibility of memory at p. Otherwise will dealloc p on destruction.
	   */
	  memory2d(index_type h, index_type w,pointer_type p, bool is_view):super_type(p,is_view),m_width(w),m_height(h) {
		  alloc();
	  }

	  /** 
	   * @brief Deallocate memory if is_view is false.
	   */
	  ~memory2d(){
		  dealloc();
	  } 

	  /*
	   * Memory Management
	   * @{
	   */

	  /** 
	   * @brief Allocate memory
	   */
	  void alloc(){
		  if (! m_is_view) {
			  cuvAssert(m_ptr == NULL);
			  m_allocator.alloc2d(&m_ptr,m_pitch,m_height,m_width);
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
	   * @brief Make an already existing memory2d a view on a raw pointer
	   *
	   * this is a substitute for operator=, when you do NOT want to copy.
	   *
	   * WARNING: ptr should not be a pitched pointer!!!
	   */
	  void set_view(index_type& pitch, const std::vector<index_type>& shape, pointer_type ptr){ 
		  dealloc();
		  m_ptr = ptr;
		  m_is_view=true;
		  set_width_and_height(shape);
		  m_pitch = pitch = m_width*sizeof(value_type);
	  }
	  /**
	   * @brief Make an already existing memory2d a view on a linear_memory.
	   *
	   * this is a substitute for operator=, when you do NOT want to copy.
	   */
	  void set_view(index_type& pitch, index_type ptr_offset, const std::vector<index_type>& shape, const linear_memory<value_type,memory_space_type,TPtr,index_type>& o){ 
		  dealloc();
		  m_ptr=o.ptr() + ptr_offset;
		  m_is_view=true;
		  set_width_and_height(shape);
		  m_pitch = pitch = m_width*sizeof(value_type);
		  cuvAssert(o.memsize() >= m_pitch * m_height + sizeof(value_type) * ptr_offset);
	  }

	  /**
	   * @brief Make an already existing memory2d a view on another memory2d.
	   *
	   * this is a substitute for operator=, when you do NOT want to copy.
	   */
	  void set_view(index_type& pitch, index_type ptr_offset, const std::vector<index_type>& shape, memory2d& o){ 
		  dealloc();
		  
		  // we can only use an offset if it does not mess up our pitching,
		  // otherwise we will get naaaasty effects
		  cuvAssert(ptr_offset*sizeof(value_type)%m_pitch == 0);
		  m_ptr=o.ptr() + ptr_offset;
		  m_is_view=true;
		  set_width_and_height(shape);
		  cuvAssert(m_width*sizeof(value_type) <= o.pitch());
		  pitch = m_pitch = o.pitch(); // keep pitch constant!
		  cuvAssert(o.memsize() >= m_pitch * m_height + sizeof(value_type) * ptr_offset);
	  }


		/** 
		 * @brief Copy memory2d.
		 * 
		 * @param o Source memory2d
		 * 
		 * @return copy to *this
		 *
		 */
	  memory2d& 
		  operator=(const memory2d& o){
			 
			if(this->size() != o.size()){
			  this->dealloc();
			  m_width  = o.m_width;
			  m_height = o.m_height;
			  this->alloc();
			}
			m_allocator.copy2d(m_ptr,o.ptr(),m_pitch,o.pitch(),m_height,m_width,memory_space_type());
			  
			return *this;
		  }

		/** 
		 * @brief Copy memory2d from other memory type.
		 * 
		 * @param o Source memory2d
		 * 
		 * @return *this
		 *
		 */
	  template<class OM, class OP>
	  memory2d& 
	  
		  operator=(const memory2d<OM, OP,index_type>& o){
			if(this->size() != o.size()){
			  this->dealloc();
			  m_width  = o.width;
			  m_height = o.height;
			  this->alloc();
			}
			m_allocator.copy2d(m_ptr,o.ptr(),m_pitch,o.pitch(),m_height,m_width,memory_space_type());
			return *this;
		  }

	  /** 
	   * @brief Assign linear memory.
	   * 
	   * @param o      Source linear memory
	   * @param oshape Source shape
	   * 
	   * @return *this
	   *
	   */
	  template<class OM, class OP>
		  memory2d& 
		  assign(const std::vector<index_type>& oshape, const linear_memory<value_type,OM,OP,index_type>& o){
			  dealloc();
			  set_width_and_height(oshape);
			  alloc();
			  m_allocator.copy2d(m_ptr,o.ptr(),m_pitch,m_width*sizeof(value_type),m_height,m_width,OM());
			  return *this;
		  }
	  /** 
	   * @brief Assign memory2d.
	   * 
	   * @param o      Source memory
	   * @param oshape Source shape
	   * 
	   * @return *this
	   *
	   */
	  template<class OM, class OP>
		  memory2d& 
		  assign(const std::vector<index_type>& oshape, const memory2d& o){
			  return this->operator=(o);
		  }

	  /**
	   * @}
	   */

	  reference_type
		  operator[](const index_type& idx)     ///< Return entry at position t
		  {
			  int x = idx % m_width;
			  int y = idx / m_width;
			  return reference_type((value_type*)((char*)m_ptr+y*m_pitch)+x);
		  }
	  const_reference_type
		  operator[](const index_type& idx)const///< Return entry at position t
		  {
			  int x = idx % m_width;
			  int y = idx / m_width;
			  return reference_type((value_type*)((char*)m_ptr+y*m_pitch)+x);
		  }

}; // memory2d


}; // cuv

#endif


