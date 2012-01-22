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
 * @file tensor.hpp
 * @brief general base class for n-dimensional matrices
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2011-03-29
 */
#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include <vector>
#include <numeric>
#include <boost/multi_array/extent_gen.hpp>
#include <boost/multi_array/index_gen.hpp>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/tools/meta_programming.hpp>
#include <cuv/basics/linear_memory.hpp>
#include <cuv/basics/reference.hpp>

namespace cuv
{
/**
 * @addtogroup basics Basic datastructures
 * @{
 */
	/// Parent struct for row and column major tags
	struct memory_layout_tag{};
	/// Tag for column major matrices
	struct column_major : public memory_layout_tag{};
	/// Tag for row major matrices
	struct row_major    : public memory_layout_tag{};

	/// converts from column to row-major and vice versa
	template<class T>
	struct other_memory_layout{};
	/// specialisation: converts from column to row-major 
	template<>
	struct other_memory_layout<column_major>{ typedef row_major type; };
	/// specialisation: converts from row to column-major 
	template<>
	struct other_memory_layout<row_major>{ typedef column_major type; };

	/// converts from dev to host memory space and vice versa
	template<class T>
	struct other_memory_space{
	};
	/// specialisation: converts from dev_memory_space to host_memory_space
	template<>
	struct other_memory_space<dev_memory_space>{ typedef host_memory_space type; };
	/// specialisation: converts from host_memory_space to dev_memory_space
	template<>
	struct other_memory_space<host_memory_space>{ typedef dev_memory_space type; };


	using boost::detail::multi_array::extent_gen;
	using boost::detail::multi_array::index_gen;
	/**
	 * defines an index range, stolen from boost::multi_array
	 *
	 * examples:
	 * @code
	 * index_range(1,3)
	 * index(1) <= index_range() < index(3)
	 * @endcode
	 */
	typedef boost::detail::multi_array::index_range<boost::detail::multi_array::index,boost::detail::multi_array::size_type> index_range;
	/**
	 * the index type used in index_range, useful for comparator syntax in @see index_range
	 */
	typedef index_range::index index;
#ifndef CUV_DONT_CREATE_EXTENTS_OBJ
	namespace{
		/**
		 * extents object, can be used to generate a multi-dimensional array conveniently.
		 *
		 * stolen from boost::multi_array.
		 *
		 * Example:
		 * @code
		 * tensor<...> v(extents[5][6][7]); // 3-dimensional tensor
		 * @endcode
		 */
		extent_gen<0> extents;
		/**
		 * indices object, can be used to generate multi-dimensional views conveniently.
		 *
		 * stolen form boost::multi_array.
		 *
		 * Example:
		 * @code
		 * tensor<...> v(indices[index_range(1,3)][index_range()], other_tensor);
		 * @endcode
		 */
		index_gen<0,0> indices;
	}
#endif

template<class V, class M, class L, class C=linear_memory_tag>
    class tensor{
        public:
            typedef typename unconst<V>::type value_type; ///< Type of contained values
            typedef M memory_space_type; ///< host or dev memory space
            typedef L memory_layout_type; ///< column_major or row_major

			typedef typename memory_traits<V,M,Tptr,index_type,__mem_container>::type  memory_container_type;     ///< the thing that allocates our storage
			typedef typename memory_container_type::reference_type reference_type;       ///< the type of the references returned by access operator
			typedef typename memory_container_type::const_reference_type const_reference_type;
			typedef typename memory_container_type::pointer_type pointer_type;  ///< type of stored pointer, could be const or not-const value_type*

    };


	/** @} */ // end group basics
}

#include <iostream>
namespace std{
	template<class V, class M, class T>
	/** 
	 * @brief Write matrix entries to stream
	 * 
	 * @param o Output stream
	 * @param w2 Matrix to output
	 */
	ostream& 
	operator<<(ostream& o, const cuv::tensor<V,M,T>& w2){
		cout << "shape=[";
                typedef typename cuv::tensor<V,M,T>::index_type I;
		typename std::vector<I>::const_iterator it=w2.shape().begin(), end=w2.shape().end();
		o <<*it;
		for(it++; it!=end; it++)
			o <<", "<<*it;
		o<<"]";
		return o;
	}
}

#endif /* __TENSOR_HPP__ */
