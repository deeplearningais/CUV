/** 
 * @file dev_vector.hpp
 * @brief vector on device
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2010-03-21
 */
#ifndef __DEV_VECTOR_HPP__
#define __DEV_VECTOR_HPP__

#include <tools/cuv_general.hpp>
#include "vector.hpp"

namespace cuv{

template<class __value_type, class __index_type=unsigned int>
class dev_vector
:    public vector<__value_type, __index_type>
{
	public:
		typedef vector<__value_type, __index_type> base_type; ///< Vector base type
		typedef dev_memory_space                   memspace_type; ///< Type of memory used: host/device
		using base_type::m_ptr;
		using typename base_type::value_type;
		using typename base_type::index_type;
	public:
		/*
		 * Construction
		 */
		dev_vector(){} ///< Calles default constructor of parent class
		/** 
		 * @brief Creates a device vector of length s and allocates memory
		 * 
		 * @param s Length of vector
		 */
		dev_vector(index_type s)
			:   base_type(s) { alloc(); }
		/** 
		 * @brief Creates device vector from device pointer
		 * 
		 * @param s Length of vector
		 * @param p Device pointer to entries
		 * @param is_view If true will not take responsibility of memory at p. Otherwise will dealloc p on destruction.
		 *
		 * Does not allocate any memory.
		 */
		dev_vector(index_type s, value_type* p, bool is_view)
			:   base_type(s,p,is_view) {} // do not alloc!
		~dev_vector(){dealloc();} ///< Deallocate memory if not a view

		/*
		 * Member access
		 */
		value_type operator[](index_type t)const; ///< Return entry at position t

		/* 
		 * Memory management
		 */
		void alloc(); ///< Allocate device memory
		void dealloc(); ///< Deallocate device memory if not a view
		void set(const index_type& i, const value_type& val); ///< Set entry i to val
};

/** 
 * @brief Trait that indicates whether device or host memory is used.
 */
template<class V, class I>
struct vector_traits<dev_vector<V,I> >{
	typedef dev_memory_space memory_space_type;  ///< Trait for memory type (host/device)

};

} // cuv

#endif

