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
#include <basics/vector.hpp>
#include <basics/matrix.hpp>
#include <tools/cuv_general.hpp>

namespace cuv{
	 /// Parent struct for row and column major tags
	struct memory_layout_tag{};
	 /// Tag for column major matrices
	struct column_major : public memory_layout_tag{};
    /// Tag for row major matrices
	struct row_major    : public memory_layout_tag{};
	
	/** 
	 * @brief Class for dense matrices
	 */
	template<class __value_type, class __mem_layout, class __memory_space_type, class __index_type = unsigned int >
	class dense_matrix 
	:        public matrix<__value_type, __index_type>{
	  public:
		  typedef __mem_layout                       				  memory_layout; 	///< Memory layout type: column_major or row_major
		  typedef __memory_space_type								  memory_space_type;///< Indicates whether matrix resides on host or device
		  typedef matrix<__value_type, __index_type>				  base_type;		///< Basic matrix type
		  typedef typename base_type::value_type 					  value_type; 		///< Type of matrix entries
		  typedef typename base_type::index_type 					  index_type;		///< Type of indices
		  typedef vector<value_type,memory_space_type,index_type>	  vec_type; 		///< Basic vector type used
		  typedef dense_matrix<value_type,memory_layout,memory_space_type,index_type>  my_type;	///< Type of this object
		  using base_type::m_width;
		  using base_type::m_height;
		  vec_type* m_vec;                      ///< stores the actual data 
		private:
		  inline const value_type operator()(const index_type& i, const index_type& j, const column_major& x) const;
		  inline const value_type operator()(const index_type& i, const index_type& j, const row_major& x)    const;
		  inline       value_type operator()(const index_type& i, const index_type& j, const column_major& x) ;
		  inline       value_type operator()(const index_type& i, const index_type& j, const row_major& x)    ;
		  inline	   void set(const index_type& i, const index_type& j, const value_type& val, const column_major&);
		  inline	   void set(const index_type& i, const index_type& j, const value_type& val, const row_major&);
		public:
		  // member access 
		  // do not return a reference, this will not work for device memory
		  inline const value_type operator()(const index_type& i, const index_type& j) const;	///< Read entry at position (i,j)
		  inline       value_type operator()(const index_type& i, const index_type& j);			///< Read entry at position (i,j)
		  inline const value_type operator()(const index_type& i) const;	///< Read entry at position (i) (vector)
		  inline       value_type operator()(const index_type& i);			///< Read entry at position (i) (vector)
		  inline size_t memsize()       const { cuvAssert(m_vec); return m_vec->memsize(); }	///< Return matrix size in memory
		  inline const value_type* ptr()const { cuvAssert(m_vec); return m_vec->ptr(); }		///< Return device pointer to matrix entries
		  inline       value_type* ptr()      { cuvAssert(m_vec); return m_vec->ptr(); }		///< Return device pointer to matrix entries
		  inline const vec_type& vec()const { return *m_vec; }									///< Return reference to vector containing matrix entries
		  inline       vec_type& vec()      { return *m_vec; }									///< Return reference to vector containing matrix entries
		  inline const vec_type* vec_ptr()const { return m_vec; }								///< Return pointer to vector containing matrix entries
		  inline       vec_type* vec_ptr()      { return m_vec; }								///< Return pointer to vector containing matrix entries
		  inline 	   void set(const index_type& i, const index_type& j, const value_type& val);///< Set entry at position (i,j)
		  inline       bool is_view()      { return m_vec->is_view(); }							///< Return true if the matrix is a view

			virtual ~dense_matrix(){
				dealloc();
			} ///< Destructor

			/** 
			 * @brief Constructor for host matrices that creates a new vector and allocates memory
			 * 
			 * @param h Height of matrix
			 * @param w Width of matrix
			 *
			 * Creates a new vector to store matrix entries.
			 */
			dense_matrix(const index_type& h, const index_type& w)
				: base_type(h,w),m_vec(NULL) {
					alloc();
				}
			/** 
			 * @brief Constructor for dense matrices that creates a matrix from a given value_type pointer
			 * 
			 * @param h Height of matrix
			 * @param w Weight of matrix 
			 * @param p Pointer to matrix entries
			 * @param is_view If true will not take responsibility of memory at p. Otherwise will dealloc p on destruction.
			 */
			dense_matrix(const index_type& h, const index_type& w, value_type* p, bool is_view = false)
				:	base_type(h,w) 
			{
				m_vec = new vec_type(h*w,p,is_view);
			}

			/** 
			 * @brief Constructor for dense matrices that creates a matrix from a given vector
			 * 
			 * @param h Height of matrix
			 * @param w Weight of matrix 
			 * @param p Pointer to a vector, storing matrix entries
			 */
			dense_matrix(const index_type& h, const index_type& w, vec_type* p)
				:	base_type(h,w),m_vec(p)
			{
			}
			
			/** 
			 * @brief Constructor for dense matrices that creates a matrix of same size as given matrix
			 * 
			 * @param m Matrix whose width and height are used to create new matrix.
			 */
		  	dense_matrix(my_type const & m)
		  	: base_type(m),
		  	  m_vec(NULL)
		  	{
				alloc();
				copy(*m_vec,m.vec());
		  	}


			/** 
			 * @brief Constructor for dense matrices that creates a matrix of same size as given matrix
			 * 
			 * @param m Matrix whose width and height are used to create new matrix.
			 */
		    template<class V, class I>
		  	dense_matrix(const matrix<V,I>* m)
		  	: base_type(m->h(),m->w()),m_vec(NULL) 
		  	{
				alloc();
		  	}

		
			void dealloc() ///< Deallocate matrix entries. This calls deallocation of the vector storing entries.
			{
				// std::cout << "Deallocate dense matrix" << std::endl;
				if(m_vec)
					delete m_vec;
				m_vec = NULL;
			}

			void alloc() ///< Allocate matrix entries: Create vector to store entries.
			{
				cuvAssert(!m_vec);
				// std::cout << "Allocate dense matrix" << std::endl;
				m_vec = new vec_type(m_width * m_height);
			}

			/** 
			 * @brief Assign value to all elements in matrix
			 * 
			 * @param scalar    the scalar value to assign
			 * 
			 * @return reference to *this
			 */
			  my_type& 
			  operator=(const value_type& scalar){
				  *m_vec = scalar;
				  return *this;
		  }
			/** 
			 * @brief Copy a matrix
			 * 
			 * @param o Source matrix
			 * 
			 * @return reference to *this
			 */
			  my_type& 
			  operator=(const my_type& o){
				  if(this==&o) return *this;
				  if(this->h() != o.h() || this->w() != o.w()){
					cuvAssert(!(m_vec->is_view())); // cannot delete view -- I do not own it!
				    this->dealloc();
				    m_width = o.w();
				    m_height = o.h();
				    this->alloc();
				  }
				  copy(*m_vec, o.vec());
				  return *this;
		  }
	};

	/*
	 * element access for dense matrix
	 *
	 */
	template<class V, class M, class T, class I>
	const typename dense_matrix<V,M,T,I>::value_type
	dense_matrix<V,M,T,I>::operator()(const index_type& i, const index_type& j, const column_major& x) const{ return (*m_vec)[ this->h()*j + i]; }

	template<class V, class M, class T, class I>
	const typename dense_matrix<V,M,T,I>::value_type
	dense_matrix<V,M,T,I>::operator()(const index_type& i, const index_type& j, const row_major& x)    const{ return (*m_vec)[ this->w()*i + j]; }

	template<class V, class M, class T, class I>
	const typename dense_matrix<V,M,T,I>::value_type
	dense_matrix<V,M,T,I>::operator()(const index_type& i, const index_type& j)    const{ return (*this)(i,j,memory_layout()); }

	template<class V, class M, class T, class I>
	typename dense_matrix<V,M,T,I>::value_type
	dense_matrix<V,M,T,I>::operator()(const index_type& i, const index_type& j, const column_major& x) { return (*m_vec)[ this->h()*j + i]; }

	template<class V, class M, class T, class I>
	typename dense_matrix<V,M,T,I>::value_type
	dense_matrix<V,M,T,I>::operator()(const index_type& i, const index_type& j, const row_major& x)    { return (*m_vec)[ this->w()*i + j]; }

	template<class V, class M, class T, class I>
	typename dense_matrix<V,M,T,I>::value_type
	dense_matrix<V,M,T,I>::operator()(const index_type& i, const index_type& j)    { return (*this)(i,j,memory_layout()); }

	template<class V, class M, class T, class I>
	typename dense_matrix<V,M,T,I>::value_type
	dense_matrix<V,M,T,I>::operator()(const index_type& i)    { return (this->vec)(i); }

	template<class V, class M, class T, class I>
	const typename dense_matrix<V,M,T,I>::value_type
	dense_matrix<V,M,T,I>::operator()(const index_type& i)const    { return (this->vec)(i); }
	/*
	 * Change values in dense matrix
	 *
	 */
	template<class V, class M, class T, class I>
	void
	dense_matrix<V,M,T,I>::set(const index_type& i, const index_type& j, const value_type& val, const column_major&) { m_vec->set( this->h()*j + i, val); };

	template<class V, class M, class T, class I>
	void
	dense_matrix<V,M,T,I>::set(const index_type& i, const index_type& j, const value_type& val, const row_major&) { m_vec->set( this->w()*i + j, val); };

	template<class V, class M, class T, class I>
	void
	dense_matrix<V,M,T,I>::set(const index_type& i, const index_type& j, const value_type& val) { this->set(i, j, val, memory_layout()); };

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
