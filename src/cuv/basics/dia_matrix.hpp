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
 * @file dia_matrix.hpp
 * @brief base class for sparse matrices in DIA format
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2010-03-21
 */
#ifndef __SPARSE_MATRIX_HPP__
#define __SPARSE_MATRIX_HPP__
#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <cuv/basics/tensor.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/basics/matrix.hpp>
#include <cuv/tools/cuv_general.hpp>

namespace cuv{
	/** 
	 * @brief Class for diagonal matrices
	 */
	template<class __value_type, class __memory_space_type, class __index_type=unsigned int> 
	class dia_matrix 
	:        public matrix<__value_type, __index_type>{
	  public:
		  typedef __value_type           value_type;	///< Type of the entries of matrix
		  typedef const value_type const_value_type;	///< Type of the entries of matrix
		  typedef matrix<__value_type, __index_type> 					   base_type; 			///< Basic matrix type
		  typedef __memory_space_type 									   memory_space_type;	///< Whether this is a host or device matrix
		  typedef typename base_type::index_type 						   index_type;			///< Type of indices
		  typedef const tensor<value_type,memory_space_type> const_vec_type; 		///< Basic vector type used
		  typedef tensor<value_type,memory_space_type>  		   vec_type; 			///< Basic vector type used
		  typedef tensor<int,memory_space_type> 				   intvec_type; 		///< Type of offsets for diagonals
		  typedef dia_matrix<value_type,memory_space_type,index_type> 	   my_type;				///< Type of this matix
		  typedef typename vec_type::pointer_type ptr_type;
		public:
		  int m_num_dia;                        ///< number of diagonals stored
		  unsigned int m_stride;                ///< how long the stored diagonals are
		  vec_type* m_vec;                      ///< stores the actual data 
		  intvec_type m_offsets;                ///< stores the offsets of the diagonals
		  std::map<int,index_type> m_dia2off;   ///< maps a diagonal to an offset
		  int m_row_fact;                       ///< factor by which to multiply a row index (allows matrices with "steep" diagonals)
		public:
		  	~dia_matrix() { ///< Destructor. Deallocates Matrix.
				dealloc();
			}

			dia_matrix() ///< Empty constructor. Returns empty diagonal matrix.
				: base_type(0,0),
				 m_vec(0),
				 m_num_dia(0),
				 m_stride(0),
				 m_row_fact(0){}
			/** 
			 * @brief Creates diagonal matrix of given size, with given number of diagonals and stride.
			 * 
			 * @param h Height of matrix 
			 * @param w Width of matrix
			 * @param num_dia Number of diagonals in matrix
			 * @param stride Stride of matrix
			 * @param row_fact Steepness of diagonals. Only 1 is supported at the moment.
			 */
			dia_matrix(const index_type& h, const index_type& w, const int& num_dia, const int& stride, int row_fact=1)
				: base_type(h,w)
				, m_vec(NULL)
				, m_num_dia(num_dia)
				, m_stride(stride)
				, m_offsets(num_dia)
			{
				m_row_fact = row_fact;
				cuvAssert(m_row_fact>0);
				alloc();

			}
			void dealloc() ///< Deallocate matrix entries. This calls deallocation of the vector storing entries.
			{
				if(m_vec){
					delete m_vec;
					}
				m_vec = NULL;
			}
			void alloc() ///< Allocate matrix entries: Create vector to store entries.
			{
				cuvAssert(m_stride >= this->h() || m_stride >= this->w());
				m_vec = new vec_type(m_stride * m_num_dia);
			}
			inline const vec_type& vec()const{ return *m_vec; } ///< Return pointer to vector storing entries
			inline       vec_type& vec()     { return *m_vec; } ///< Return pointer to vector storing entries
			inline const vec_type* vec_ptr()const{ return m_vec; } ///< Return reference to vector storing entries
			inline       vec_type* vec_ptr()     { return m_vec; } ///< Return reference to vector storing entries
			inline       ptr_type ptr()     { return m_vec->ptr(); } ///< Return reference to vector storing entries
			inline const ptr_type ptr()const{ return m_vec->ptr(); } ///< Return reference to vector storing entries
			inline int num_dia()const{ return m_num_dia; } ///< Return number of diagonals
			inline unsigned int stride()const { return m_stride;  }///< Return stride of matrix
			inline int row_fact()const{ return m_row_fact; } ///< Return steepness of diagonals

			//*****************************
			// set/get offsets of diagonals
			//*****************************
		
			/**
			 * Set the offsets of the diagonals in the DIA matrix
			 * This overload works with any iterator.
			 * The main diagonal has offset zero, lower diagonals are negative, higher diagonals are positive.
			 * 
			 * @param begin start of sequence
			 * @param end   one behind end of sequence
			 */
			template<class T>
			void set_offsets(T begin, const T& end){ 
				int i=0;
				while(begin!=end)
					m_offsets[i++]= *begin++;
				post_update_offsets();
			}
			/**
			 * Set the offsets of the diagonals in the DIA matrix.
			 * This overload works with a vector.
			 * The main diagonal has offset zero, lower diagonals are negative, higher diagonals are positive.
			 *
			 * @param v a vector containing all offsets
			 */
			template<class T>
			void set_offsets(const std::vector<T>& v){
				for(unsigned int i=0;i<v.size();i++)
					m_offsets[i]=v[i];
				post_update_offsets();
			}
			/**
			 * Update the internal reverse-mapping of diagonals to positions in the offset-array.
			 * Normally, you do not need to call this function, except when
			 * changing the diagonal offsets manually.
			 */
			void post_update_offsets(){
				m_dia2off.clear();
				for(unsigned int i = 0; i<m_offsets.size(); ++i)
					m_dia2off[m_offsets[i]] = i;
			}
			/**
			 * Set a single diagonal offset in the matrix.
			 * The main diagonal has offset zero, lower diagonals are negative, higher diagonals are positive.
			 *
			 * @param idx the (internal) index of the offset
			 * @param val the offset number
			 */
			inline void set_offset(const index_type& idx, const index_type& val){
				m_offsets[idx] = val;
				m_dia2off[val] = idx;
			}
			/** Return vector (view) containing specified diagonal.
			 */
			inline const_vec_type* get_dia(const int& k)const{ 
				typename std::map<int,index_type>::const_iterator it = m_dia2off.find(k);
				int off = it->second;
				const index_type i_start = std::max((int)0,-k);
				const index_type j_start = std::max((int)0, k);
				const index_type N = std::min(base_type::m_height - i_start, (base_type::m_width - j_start));
				return new const_vec_type(extents[m_stride],  m_vec->ptr() + off * m_stride); 
				//return new vec_type(N,  m_vec->ptr() + off * m_stride + i_start, true); 
			} 
			/** Return a vector (view) on the specified diagonal
			 */
			inline vec_type* get_dia(const int& k){ 
				int off   = m_dia2off[k];
				const index_type i_start = std::max((int)0,-k);
				const index_type j_start = std::max((int)0, k);
				const index_type N = std::min(base_type::m_height - i_start, (base_type::m_width - j_start));
				return new vec_type(extents[m_stride],  m_vec->ptr() + off * m_stride); 
				//return new vec_type(N,  m_vec->ptr() + off * m_stride + i_start, true); 
			} 
			inline const intvec_type& get_offsets()const{return m_offsets;} ///< Return the vector of offsets
			inline       intvec_type& get_offsets()     {return m_offsets;} ///< return the vector of offsets
			inline int get_offset(const index_type& idx)const               ///< Return offset of specified diagonal
			{
				return m_offsets[idx];
			}

			// ******************************
			// read access
			// ******************************
			void set(const index_type& i, const index_type& j, const value_type& val)
			{
				int off = (int)j - (int)i/m_row_fact;
				typename std::map<int,index_type>::const_iterator it = m_dia2off.find(off);
				if( it == m_dia2off.end() )
					return;
				(*m_vec)[it->second * m_stride +i]=val;
			}
			value_type operator()(const index_type& i, const index_type& j)const ///< Return matrix entry (i,j)
			{
				int off = (int)j - (int)i/m_row_fact;
				typename std::map<int,index_type>::const_iterator it = m_dia2off.find(off);
				if( it == m_dia2off.end() )
					return (value_type) 0;
				return (*m_vec)[ it->second * m_stride +i  ];
			}
			bool has(const index_type& i, const index_type& j)const ///< Return whether matrix entry is managed by sparse matrix 
			{
				int off = (int)j - (int)i/m_row_fact;
				typename std::map<int,index_type>::const_iterator it = m_dia2off.find(off);
				if( it == m_dia2off.end() )
					return false;
				return true;
			}
			/** 
			 * @brief Assignment operator. Assigns vector belonging to source to destination and sets source vector to NULL
			 * 
			 * @param o Source matrix
			 * 
			 * @return Matrix of same size and type of o that now owns vector of entries of o.
			 */
		  my_type& 
			  operator=(const my_type& o){
				  if(this==&o) return *this;
				  this->dealloc();

				  (base_type&) (*this)  = (base_type&) o; 
				  m_vec=o.m_vec;
				  m_num_dia = o.m_num_dia;
				  m_stride = o.m_stride;
				  m_row_fact = o.m_row_fact;
				  m_offsets = o.m_offsets;
				  m_dia2off = o.m_dia2off;

				   // transfer ownership of memory (!)
				  (const_cast< my_type *>(&o))->m_vec = NULL;
				  return *this;
			  }
	};
}

namespace std{
	template<class V, class T, class I>
	/** 
	 * @brief Return stream containing matrix entries for debugging
	 * 
	 * @param o Output stream
	 * @param w2 Matrix to output
	 */
	ostream& 
	operator<<(ostream& o, const cuv::dia_matrix<V,T,I>& w2){
		cout << "Dia-Matrix: "<<endl;
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

#endif /* __SPARSE_MATRIX_HPP__ */

