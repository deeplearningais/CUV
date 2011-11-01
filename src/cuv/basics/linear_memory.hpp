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
#include <vector>
#include <numeric>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/tools/meta_programming.hpp>
#include <cuv/basics/reference.hpp>
#include <cuv/basics/memory.hpp>
#include <cuv/basics/memory2d.hpp>
#include <boost/foreach.hpp>

namespace cuv{

/**
 * @brief Basic linear memory class
 *
 * This linear memory class is the generic storage classes and is repsonsible for allocation/deletion.
 */
template<class __value_type, class __memory_space_type, class TPtr=const __value_type*, class __index_type=unsigned int>
class linear_memory
: public memory<__value_type,__memory_space_type,TPtr,__index_type>
{
	public:
	  typedef memory<__value_type,__memory_space_type,TPtr,__index_type> super_type; ///< Base class we're deriving from

	  typedef typename super_type::value_type value_type; ///< Type of the entries of memory
	  typedef typename super_type::const_value_type const_value_type; ///< Type of the entries of memory
	  typedef typename super_type::index_type index_type; ///< Type indices/dimensions
	  typedef typename super_type::memory_space_type memory_space_type; ///< Indicates whether this is a host or device vector
	  typedef typename super_type::pointer_type      pointer_type; ///< Type of stored pointer, should be const or not-const value_type*
	  typedef typename super_type::reference_type      reference_type; ///< Type of references returned by operator[]
	  typedef typename super_type::const_reference_type      const_reference_type; ///< Type of references returned by operator[]
	  typedef linear_memory<value_type, memory_space_type, TPtr, index_type> my_type; ///< Type of this linear_memory
	  typedef memory2d<value_type, memory_space_type, TPtr, index_type> mem2d_type; ///< Type of 2d memory with similar properties
	protected:
	  using super_type::m_ptr;      ///< pointer to stored data
	  using super_type::m_is_view;  ///< whether we own the data in m_ptr
	  using super_type::m_allocator; ///< how data was allocated
	  index_type m_size;            ///< number of elements in data

	  /**
	   * get size and set pitch
	   * @param pitch OUT pitch of the resulting memory
	   * @param shape shape of the desired memory
	   * @param inner_is_last whether the last component of shape denotes the inner-most=pitched dimension
	   */
	  index_type get_size_pitch(index_type& pitch, const std::vector<index_type>& shape, bool inner_is_last){
		  pitch = sizeof(value_type) * (inner_is_last ? shape.back() : shape.front());
		  return std::accumulate(shape.begin(),shape.end(),(index_type)1,std::multiplies<index_type>());
	  }

	public:
	  /*
	   * Member Access
	   */

	  /**
	   * @brief Return length of linear_memory
	   */
	  inline index_type size() const         { return m_size; }
	  /**
	   * @brief set length of linear_memory. If this is longer than the
	   * original size, this will destroy the content due to reallocation!
	   *
	   * @param pitch OUT pitch of the resulting memory
	   * @param shape shape of the desired memory
	   * @param inner_is_last whether the last component of shape denotes the inner-most=pitched dimension
	   */
	  void set_size(index_type& pitch, const std::vector<index_type>& shape, bool inner_is_last) {
		  int s = get_size_pitch(pitch,shape,inner_is_last);
		  if(m_size < s){
			  dealloc();
			  m_size = s;
			  m_is_view = false;
			  alloc();
			  return;
		  }
		  m_size    = s;
	  }
	  /*
	   * Construction
	   */
	  /** 
	   * @brief Empty constructor. Creates empty linear_memory (allocates no memory)
	   */
	  explicit linear_memory():m_size(0){} 
	  /** 
	   * @brief Creates linear_memory of lenght s and allocates memory
	   * 
	   * @param s Length of linear_memory
	   */
	  explicit linear_memory(index_type s):m_size(s) {
		  alloc();
	  }
	  /** 
	   * @brief Copy-Constructor
	   * @param o source
	   */
	  explicit linear_memory(const my_type& o):m_size(o.size()) {
		  alloc();
		  m_allocator.copy(this->m_ptr,o.ptr(),size(),memory_space_type());
	  }
	  /** 
	   * @brief Copy-Constructor for other linear memory spaces
	   * @param o source
	   */
	  template<class OM, class OP>
	  explicit linear_memory(const linear_memory<__value_type, OM, OP,__index_type>& o):m_size(o.size()) {
		  alloc();
		  m_allocator.copy(this->m_ptr,o.ptr(),size(),OM());
	  }
	  /** 
	   * @brief Copy-Constructor for other memory spaces
	   * @param o source
	   */
	  template<class OM, class OP>
	  explicit linear_memory(const memory2d<__value_type, OM, OP,__index_type>& o):m_size(o.width()*o.height()) {
		  alloc();
		  m_allocator.copy2d(this->m_ptr,o.ptr(),o.width()*sizeof(value_type),o.pitch(),o.height(),o.width(),OM());
	  }
	  /** 
	   * @brief Creates linear_memory from pointer to entries.
	   * 
	   * @param s Length of linear_memory
	   * @param p Pointer to entries 
	   * @param is_view If true will not take responsibility of memory at p. Otherwise will dealloc p on destruction.
	   */
	  explicit linear_memory(index_type s,pointer_type p, bool is_view):super_type(p,is_view),m_size(s) {
		  alloc();
	  }

	  /**
	   * @brief Make an already existing linear_memory a view on another linear_memory.
	   *
	   * this is a substitute for operator=, when you do NOT want to copy.
	   *
	   * @warning ptr should not be a pitched pointer!!!
	   *
	   * @param pitch OUT pitch of the resulting memory
	   * @param ptr_offset IN offset relative to o.ptr()
	   * @param shape shape of the desired memory
	   * @param p Pointer to entries 
	   * @param inner_is_last whether the last component of shape denotes the inner-most=pitched dimension
	   */
	  void set_view(index_type& pitch, index_type ptr_offset, const std::vector<index_type>& shape, pointer_type p, bool inner_is_last){ 
		  dealloc();
		  this->m_ptr     = p+ptr_offset;
		  m_is_view = true;
		  m_size    = get_size_pitch(pitch,shape,inner_is_last);
	  }

	  /**
	   * @overload
	   * @brief Make an already existing linear_memory a view on another linear_memory.
	   *
	   * this is a substitute for operator=, when you do NOT want to copy.
	   *
	   * @param pitch OUT pitch of the resulting memory
	   * @param ptr_offset IN offset relative to o.ptr()
	   * @param shape shape of the desired memory
	   * @param o source
	   * @param inner_is_last whether the last component of shape denotes the inner-most=pitched dimension
	   */
	  void set_view(index_type& pitch, index_type ptr_offset, const std::vector<index_type>& shape, const linear_memory& o, bool inner_is_last ){ 
		  dealloc();
		  this->m_ptr=o.ptr() + ptr_offset;
		  m_is_view=true;
		  m_size    = get_size_pitch(pitch,shape,inner_is_last);
		  cuvAssert(o.size()>= m_size + ptr_offset);
	  }
	  /**
	   * @overload
	   * @brief Make an already existing linear_memory a view on a memory2d.
	   *
	   * this is a substitute for operator=, when you do NOT want to copy.
	   *
	   * @param pitch OUT pitch of the resulting memory
	   * @param ptr_offset IN offset relative to o.ptr()
	   * @param shape shape of the desired memory
	   * @param o source
	   * @param inner_is_last whether the last component of shape denotes the inner-most=pitched dimension
	   */
	  void set_view(index_type& pitch, index_type ptr_offset, const std::vector<index_type>& shape, const memory2d<value_type, memory_space_type, TPtr,index_type>& o, bool inner_is_last){ 
		  dealloc();
		  if(ptr_offset != 0){
			  // we can only use an offset if it does not mess up our pitching,
			  // otherwise we will get naaaasty effects
			  int x = ptr_offset % o.width();
			  int y = ptr_offset / o.width();
			  //cuvAssert((y*o.pitch()+x*sizeof(value_type))%o.pitch() == 0);
			  this->m_ptr=(value_type*)((char*)o.ptr() + y * o.pitch())+ x;
		  }else{
			  this->m_ptr = o.ptr();
		  }
		  m_is_view=true;
		  m_size    = get_size_pitch(pitch,shape,inner_is_last);
		  cuvAssert(o.memsize()>= memsize() + sizeof(value_type)*ptr_offset);

		  // make sure the memory2d is unpitched or we just view one line
		  cuvAssert( o.width()*sizeof(value_type) <= memsize() 
			||   o.pitch()*o.height() == o.width()*o.height()*sizeof(value_type));
	  }

	  /** 
	   * @brief Deallocate memory if is_view is false.
	   */
	  ~linear_memory(){
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
			  cuvAssert(this->m_ptr == NULL)
			  m_allocator.alloc( &this->m_ptr,m_size);
		  }
	  } 

	  /** 
	   * @brief Return size of linear_memory in memory
	   */
	  inline size_t memsize()       const{ return m_size*sizeof(value_type); } 


	  /** 
	   * @brief Deallocate memory if not a view
	   */
	  void dealloc(){
		  if (this->m_ptr && ! m_is_view)
			m_allocator.dealloc(&this->m_ptr);
		  this->m_ptr=NULL;
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
			  m_allocator.copy(this->m_ptr,o.ptr(),size(),memory_space_type());

			  return *this;
		  }

	  /** 
	   * @overload
	   *
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
			  m_allocator.copy(this->m_ptr,o.ptr(),size(),OM());
			  return *this;
		  }

	  /** 
	   * @overload
	   *
	   * @brief Copy linear_memory from other value type
	   * 
	   * @param o Source linear_memory
	   * 
	   * @return copy to *this
	   *
	   * @internal
	   * We need the DisableIf here, since the compiler otherwise
	   * cannot know which version of operator= to call when both
	   * value_type and memory_space_type are the same.
	   *
	   */
	  template<class OV, class OP>
		  typename EnableIf<typename IsDifferent<OV,value_type>::Result,my_type>::type&
		  operator=(const linear_memory<OV, memory_space_type, OP,index_type>& o){
			  if(this->size() != o.size()){
				  this->dealloc();
				  m_size = o.size();
				  this->alloc();
			  }
			  m_allocator.copy(this->m_ptr,o.ptr(),size(),memory_space_type());
			  return *this;
		  }

	  /** 
	   * @overload
	   *
	   * @brief Copy linear_memory from memory2d type.
	   * 
	   * @param o Source linear_memory
	   * 
	   * @return copy to *this
	   *
	   */
	  template<class OM, class OP>
		  my_type&
		  operator=(const memory2d<value_type, OM, OP,index_type>& o){
			  dealloc();
			  m_size = o.width()*o.height();
			  alloc();
			  m_allocator.copy2d(this->m_ptr,o.ptr(),o.width()*sizeof(value_type),o.pitch(),o.height(),o.width(),OM());
			  return *this;
		  }

	  /**
	   * assign another linear memory (shape parameter is ignored)
	   *
	   * @param oshape    shape of o, ignored
	   * @param o         source linear memory
	   */
	  template<class OM, class OP>
		  my_type&
		  assign(index_type& pitch, const std::vector<index_type>& oshape, const linear_memory<value_type,OM,OP,index_type>& o, bool inner_is_last ){
			  operator=(o);
			  get_size_pitch(pitch,oshape,inner_is_last);
			  return *this;
		  }

	  /**
	   * @overload
	   * assign another linear memory (shape parameter is ignored)
	   *
	   * @param oshape    shape of o, ignored
	   * @param o         source linear memory
	   *
	   * @internal
	   * We need the DisableIf here, since the compiler otherwise
	   * cannot know which version of assign to call when both
	   * value_type and memory_space_type are the same.
	   */
	  template<class OV, class OP>
		  typename EnableIf<typename IsDifferent<OV,value_type>::Result,my_type>::type&
		  assign(index_type& pitch, const std::vector<index_type>& oshape, const linear_memory<OV,memory_space_type,OP,index_type>& o, bool inner_is_last ){
			  operator=(o);
			  get_size_pitch(pitch,oshape,inner_is_last);
			  return *this;
		  }

	  /**
	   * @overload
	   *
	   * assign memory2d (shape parameter is ignored)
	   *
	   * @param oshape    shape of o, ignored, instead width/height/pitch of o is used
	   * @param o         source linear memory
	   */
	  template<class OM, class OP>
		  my_type&
		  assign(index_type& pitch, const std::vector<index_type>& oshape, const memory2d<value_type,OM,OP,index_type>& o, bool inner_is_last ){
			  operator=(o);
			  get_size_pitch(pitch,oshape,inner_is_last);
			  return *this;
		  }

	  /**
	   * @overload
	   *
	   * assign memory2d (shape parameter is ignored)
	   *
	   * @param oshape    shape of o, ignored, instead width/height/pitch of o is used
	   * @param o         source linear memory
	   *
	   * @internal
	   * We need the DisableIf here, since the compiler otherwise
	   * cannot know which version of assign to call when both
	   * value_type and memory_space_type are the same.
	   */
	  template<class OV, class OP>
		  typename EnableIf<typename IsDifferent<OV,value_type>::Result,my_type>::type&
		  assign(index_type& pitch, const std::vector<index_type>& oshape, const memory2d<OV,memory_space_type,OP,index_type>& o, bool inner_is_last ){
			  operator=(o);
			  get_size_pitch(pitch,oshape,inner_is_last);
			  return *this;
		  }

	  reference_type
		  operator[](const index_type& idx)     ///< Return entry at position t
		  {
			  return reference_type(this->m_ptr+idx);
		  }
	  const_reference_type
		  operator[](const index_type& idx)const///< Return entry at position t
		  {
			  return const_reference_type(this->m_ptr+idx);
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


struct linear_memory_tag{};
template<class V,class M, class P, class I>
struct memory_traits<V,M,P,I,linear_memory_tag>{
	typedef linear_memory<V,M,P,I> type;
};

}; // cuv

#endif

