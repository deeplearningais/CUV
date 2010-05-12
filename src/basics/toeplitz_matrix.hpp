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
 * @file toeplitz_matrix.hpp
 * @brief base class for sparse matrices in DIA format
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2010-05-10
 */
#ifndef __TOEPLITZ_MATRIX_HPP__
#define __TOEPLITZ_MATRIX_HPP__
#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <vector.hpp>
#include <vector_ops/vector_ops.hpp>
#include <matrix.hpp>
#include <cuv_general.hpp>

namespace cuv{
	/** 
	 * @brief Class for diagonal toeplitz-matrices
	 */
	template<class __value_type, class __memory_space_type, class __index_type=unsigned int> 
	class toeplitz_matrix 
	:        public matrix<__value_type, __index_type>{
	  public:
		  typedef matrix<__value_type, __index_type> 					   base_type; 			///< Basic matrix type
		  typedef __memory_space_type 									   memory_space_type;	///< Whether this is a host or device matrix
		  typedef typename base_type::value_type 						   value_type;			///< Type of matrix entries
		  typedef typename base_type::index_type 						   index_type;			///< Type of indices
		  typedef vector<value_type,memory_space_type,index_type>  		   vec_type; 			///< Basic vector type used
		  typedef vector<int,memory_space_type,index_type> 				   intvec_type; 		///< Type of offsets for diagonals
		  typedef toeplitz_matrix<value_type,memory_space_type,index_type> 	   my_type;				///< Type of this matix
		public:
		  int m_num_dia;                        ///< number of diagonals stored
		  vec_type* m_vec;                      ///< stores the actual data 
		  intvec_type m_offsets;                ///< stores the offsets of the diagonals
		  std::map<int,index_type> m_dia2off;   ///< maps a diagonal to an offset
		  int m_input_maps;                     ///< number of input maps  (along 1st dimension)
		  int m_output_maps;                    ///< number of output maps (along 2nd dimension)
		public:
		  	~toeplitz_matrix() { ///< Destructor. Deallocates Matrix.
				dealloc();
			}

			toeplitz_matrix() ///< Empty constructor. Returns empty diagonal matrix.
				: base_type(0,0),
				 m_vec(0),
				 m_num_dia(0),
				 m_input_maps(1),
				 m_output_maps(1)
				 {}
			/** 
			 * @brief Creates diagonal matrix of given size, with given number of diagonals.
			 * 
			 * @param h Height of matrix 
			 * @param w Width of matrix
			 * @param num_dia Number of diagonals in matrix
			 * @param input_maps number of input maps (along 1st dim of matrix)
			 * @param output_maps number of output maps (along 2nd dim of matrix)
			 */
			toeplitz_matrix(const index_type& h, const index_type& w, const int& num_dia, const int& input_maps=1, const int& output_maps=1)
				: base_type(h,w)
				, m_num_dia(num_dia)
				, m_offsets(num_dia)
				, m_input_maps(input_maps)
				, m_output_maps(output_maps)
			{
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
				m_vec = new vec_type(m_num_dia*m_input_maps);
			}
			inline int input_maps()const{ return m_input_maps; } ///< Return number of input_maps
			inline int output_maps()const{ return m_output_maps; } ///< Return number of output
			inline const vec_type& vec()const{ return *m_vec; } ///< Return pointer to vector storing entries
			inline       vec_type& vec()     { return *m_vec; } ///< Return pointer to vector storing entries
			inline const vec_type* vec_ptr()const{ return m_vec; } ///< Return reference to vector storing entries
			inline       vec_type* vec_ptr()     { return m_vec; } ///< Return reference to vector storing entries
			inline int num_dia()const{ return m_num_dia; } ///< Return number of diagonals

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
					m_offsets.set(i++,*begin++);
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
					m_offsets.set(i,v[i]);
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
				m_offsets.set(idx,val);
				m_dia2off[val] = idx;
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
			value_type operator()(const index_type& i, const index_type& j)const ///< Return matrix entry (i,j)
			{
				int off = (int)j - (int)i;
				typename std::map<int,index_type>::const_iterator it = m_dia2off.find(off);
				if( it == m_dia2off.end() )
					return (value_type) 0;
				int w = base_type::m_width / m_output_maps;
				int p = 2*( off<=0 )-1;
				int z = off + p*( int( -p* off/float(w) + 0.5f)*w );
				float elim = !( 
						   (z> 0 && (j%w)<z  ) 
						|| (z<=0 && (j%w)>=w+z) );
				return elim * (*m_vec)[ it->second*m_input_maps + j/w];
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
	operator<<(ostream& o, const cuv::toeplitz_matrix<V,T,I>& w2){
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

#endif /* __TOEPLITZ_MATRIX_HPP__ */

