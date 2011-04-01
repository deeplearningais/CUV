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
#include <iostream>
#include <cuv/basics/tensor.hpp>
#include <cuv/tools/cuv_general.hpp>
//#include <cuv/vector_ops/vector_ops.hpp>
#include <cuv/basics/accessors.hpp>

namespace cuv{


/**
 * @brief Basic vector class
 *
 * This vector class is the parent of all other vector classes and has all the basic attributes that all matrices share.
 * This class is never actually instanciated.
 */
template<class __value_type, class __memory_space_type, class __index_type=unsigned int>
class vector
:public tensor<__value_type,column_major,__memory_space_type>{
	public:
		typedef tensor<__value_type,column_major,__memory_space_type> tensor_type;
		typedef typename tensor_type::value_type         value_type;
		typedef typename tensor_type::memory_space_type  memory_space_type;
		typedef typename tensor_type::memory_layout_type memory_layout_type;
		typedef typename tensor_type::index_type         index_type;
		typedef typename tensor_type::pointer_type       pointer_type;
	
        /// default constructor does nothing (especially no memory allocation).
	vector(){ }

        /// we simply add a convenience constructor here.
	vector(index_type l)
		:tensor_type(extents[l]){
		}
        /// we simply add a convenience constructor here.
	vector(const index_type& l, pointer_type ptr, bool b=true)
		:tensor_type(extents[l], ptr){
		}
	/// deprecated! use the reference returned by []
	void set(const index_type& idx, const value_type& val){
		std::cout << " DEPRECATED SET IN VEC" <<std::endl;
		(*this)[idx] = val;
	}
};

}; // cuv

#endif
