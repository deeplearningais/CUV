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
 * @file memory.hpp
 * @brief base class for linear_memory, two_d_memory, ...
 * @ingroup basics
 * @author Hannes Schulz
 */

#ifndef __MEMORY_HPP__
#define __MEMORY_HPP__
#include <vector>
#include <cuv/basics/reference.hpp>

namespace cuv
{


	/**
	 * @brief Abstract base class of all memory classes
	 */

	template <class __value_type,
		 class __memory_space_type,
		 class TPtr=const __value_type*,
		 class __index_type=unsigned int
		 >
	class memory
	{
	public:
		typedef __value_type                value_type;        ///< Type of the entries of memory
		typedef const __value_type    const_value_type;        ///< Type of the entries of memory
		typedef __index_type        index_type;        ///< Type of indices
		typedef __memory_space_type memory_space_type; ///< Indicates whether this is a host or device vector
		typedef TPtr                pointer_type;      ///< Type of stored pointer, should be const or not-const char*
		typedef reference<value_type, memory_space_type,index_type> reference_type; ///< Type of references returned by operators
		typedef const reference_type const_reference_type; ///< Type of references returned by operators
		allocator<value_type, index_type, memory_space_type> m_allocator;

		memory() :m_ptr(NULL),m_is_view(false){}
		memory(pointer_type p, bool is_view) :m_ptr(p),m_is_view(is_view){}


		/* ****************************
		 * Member access
		 * ****************************/

		/**
		 * @brief Return pointer to entries
		 */
		inline const pointer_type ptr()const{ return m_ptr; }

		/**
		 * @brief Return pointer to entries
		 */
		inline pointer_type ptr()           { return m_ptr; }

		/** 
		 * @brief Return true if this memory is a view and should not call free
		 */
		inline bool is_view() const         { return m_is_view; }

	protected:
		pointer_type m_ptr; ///< Pointer to actual entries in memory
		bool         m_is_view; ///< Indicates whether this linear_memory owns the memory or is just a view
	};
	

	template<class V, class M, class P, class I, class MemoryContainer>
	struct memory_traits{
		typedef memory<V,M,P,I> type;
	};



}
#endif /* __MEMORY_HPP__ */
