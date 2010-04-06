/** 
 * @file host_vector.hpp
 * @brief vector on host
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2010-03-21
 */
#ifndef __HOST_VECTOR_HPP__
#define __HOST_VECTOR_HPP__

#include <tools/cuv_general.hpp>
#include "vector.hpp"

namespace cuv{

template<class __value_type, class __index_type=unsigned int>
class host_vector
:    public vector<__value_type, __index_type>
{
	public:
		typedef vector<__value_type, __index_type> base_type; ///< Vector base type
		typedef host_memory_space                  memspace_type; ///< Type of memory used: host/device
		using typename base_type::value_type;
		using typename base_type::index_type;
		using base_type::m_ptr;
		using base_type::m_is_view;
	public:
		/*
		 * Construction
		 */

		host_vector() ///< Creates host vector of size 0
			:   base_type(0) {}
		host_vector(index_type s)///< Creates host vector of length s and allocates memory
			:   base_type(s) { alloc(); }
		/** 
		 * @brief Creates host vector from pointer
		 * 
		 * @param s Length of vector
		 * @param p Device pointer to entries
		 * @param is_view If true will not take responsibility of memory at p. Otherwise will dealloc p on destruction.
		 *
		 * Does not allocate any memory.
		 */
		host_vector(index_type s, value_type* p, bool is_view)
			:   base_type(s,p,is_view) { } // do not alloc!
		~host_vector(){ dealloc(); } ///< Deallocate memory if not a view

		/*
		 * Member access
		 */
		inline value_type operator[](const index_type& idx)const{ return m_ptr[idx]; }///< Return entry at position t

		/*
		 * Memory Management
		 */
		void alloc() ///< Allocate host memory
		{
			m_ptr = new value_type[this->size()];
		}
		void dealloc() ///< Deallocate host memory if not a view
		{
			if(m_ptr && !m_is_view){
				delete[] m_ptr;
			}
			m_ptr = NULL;
		}

		void set(const index_type& i, const value_type& val) ///< Set entry i to val
			{ m_ptr[i] = val; }
};

/** 
 * @brief Trait that indicates whether device or host memory is used.
 */
template<class V, class I>
struct vector_traits<host_vector<V,I> >{
	typedef dev_memory_space memory_space_type;
};

} // cuv

#endif
