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
 * @file dense_matrix.hpp
 * @brief base class for dence matrices
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2010-03-21
 */
#ifndef __DENSE_MATRIX_HPP__
#define __DENSE_MATRIX_HPP__
#include <cuv/basics/tensor.hpp>
#include <cuv/basics/linear_memory.hpp>
#include <cuv/tools/cuv_general.hpp>

namespace cuv{
	/** 
	 * @brief Class for dense matrices
	 */
	template<class __value_type, class __memory_space_type, class __mem_layout = row_major, class __index_type = unsigned int >
	class dense_matrix 
	:public tensor<__value_type,__memory_space_type, __mem_layout>{
		public:
		typedef tensor<__value_type,__memory_space_type,__mem_layout> tensor_type;
		typedef typename tensor_type::value_type         value_type;
		typedef typename tensor_type::const_value_type   const_value_type;
		typedef typename tensor_type::memory_space_type  memory_space_type;
		typedef typename tensor_type::memory_layout_type memory_layout;
		typedef typename tensor_type::index_type         index_type;
		typedef typename tensor_type::pointer_type       pointer_type;
		//typedef linear_memory<value_type,memory_space_type>     vector_type;
		//private:
		//mutable vector_type m_vec;
		public:
		using tensor_type::operator=;
		
		/// we simply add a convenience constructor here.
		dense_matrix(const index_type& h, const index_type& w)
			:tensor_type(extents[h][w]){
			}

		/// we simply add a convenience constructor here.
		dense_matrix(const index_type& h, const index_type& w, const pointer_type ptr, bool is_view=true)
			:tensor_type(extents[h][w],ptr){
			}

		/// Copy constructor
		dense_matrix(const dense_matrix& o)
			:tensor_type(o)
		{
		}

		/// Copy constructor for other memory spaces
		template<class MO, class ML>
		dense_matrix(const dense_matrix<__value_type,MO,ML,__index_type>& o)
			:tensor_type(o)
		{
		}

		/// deprecated! use the reference returned by (..)
		//void set(const index_type& i, const index_type& j, const value_type& val){
			//std::cout << "DEPRECATED SET"<<std::endl;
			//(*this)(i,j) = val;
		//}
		const index_type& h()const{ return this->shape()[0]; };
		const index_type& w()const{ return this->shape()[1]; };
		const index_type  n()const{ return this->size(); };


		///// deprecated! define your stuff on tensor instead!
		//linear_memory<value_type,memory_space_type>&
		//vec(){
			//std::cout << "DEPRECATED VEC"<<std::endl;
			//m_vec = vector<value_type,memory_space_type>(this->size(),this->ptr());
			//return m_vec;
		//}

		///// deprecated! define your stuff on tensor instead!
		//const vector<value_type,memory_space_type>&
		//vec()const{
			//std::cout << "DEPRECATED VEC"<<std::endl;
			//m_vec = vector<value_type,memory_space_type>(this->size(),this->ptr());
			//return m_vec;
		//}
		/// deprecated! use ptr()
		const_value_type* vec_ptr()const{
			std::cout << "DEPRECATED vec_ptr"<<std::endl;
			return this->ptr();}
		/// deprecated! use ptr()
		value_type*       vec_ptr()     {
			std::cout << "DEPRECATED vec_ptr"<<std::endl;
			return this->ptr();}
	};

	template<class Mat, class NewVT>
		struct switch_value_type{
			typedef dense_matrix<NewVT, typename Mat::memory_layout, typename Mat::memory_space_type, typename Mat::index_type> type;
		};
	template<class Mat, class NewML>
		struct switch_memory_layout_type{
			typedef dense_matrix<typename Mat::value_type, NewML, typename Mat::memory_space_type, typename Mat::index_type> type;
		};
	template<class Mat, class NewMS>
		struct switch_memory_space_type{
			typedef dense_matrix<typename Mat::value_type, typename Mat::memory_layout, NewMS, typename Mat::index_type> type;
		};
}

#include <iostream>
namespace std{
	template<class V, class M, class T, class I>
	/** 
	 * @brief Return stream containing matrix entries for debugging
	 * 
	 * @param o Output stream
	 * @param w2 Matrix to output
	 */
	ostream& 
	operator<<(ostream& o, const cuv::dense_matrix<V,M,T,I>& w2){
		cout << "Dense-Matrix: "<<endl;
		for(I i=0;i<w2.h();i++){
			for(I j=0;j<w2.w();j++){
				o << w2(i,j) << " ";
			}
			o << endl;
		}
		o << endl;
		return o;
	}
}


#endif /* __DENSE_MATRIX_HPP__ */
