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
//template<class __value_type, class __index_type>
class dev_vector
:    public vector<__value_type, __index_type>
{
	public:
		typedef vector<__value_type, __index_type> base_type; ///< Vector base type
		typedef dev_memory_space                   memspace_type; ///< Type of memory used: host/device
		using base_type::m_ptr;
		typedef typename base_type::value_type value_type;
		typedef typename base_type::index_type index_type;
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

		/*
		 * Member access
		 */
		value_type operator[](index_type idx)const; ///< Return entry at position t

		/* 
		 * Memory management
		 */
		/** 
		 * @brief Allocate device memory
		 */
		void alloc(); 
		/** 
		 * @brief Deallocate device memory if not a view
		 */
		void dealloc(); 
		/** 
		 * @brief Set entry i to val
		 * 
		 * @param i Index of which entry to change 
		 * @param val New value of entry
		 */
		void set(const index_type& i, const value_type& val); 
};

/** 
 * @brief Trait that indicates whether device or host memory is used.
 */
//template<class V, class I>
//struct vector_traits<dev_vector<V,I> >{
	//typedef dev_memory_space memory_space_type;  ///< Trait for memory type (host/device)

//};


//template<class __value_type,class __index_type>
//struct matrix_traits<__value_type, __index_type,dev_memory_space> {
	//typedef dev_vector<__value_type, __index_type>  vector_type;
//};
} // cuv

#endif

