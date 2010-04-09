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
	struct memory_layout_tag{};
	struct column_major : public memory_layout_tag{}; ///< Trait for column major matrices
	struct row_major    : public memory_layout_tag{}; ///< Trait for row major matrices
	
	//template<class __value_type,class __index_type>
	//struct matrix_traits<__value_type, __index_type,dev_memory_space> {
		//typedef dev_vector<__value_type, __index_type>  vector_type;
	//};

	//template<class __value_type,class __index_type>
	//struct matrix_traits<__value_type, __index_type,host_memory_space> {
		//typedef host_vector<__value_type, __index_type>  vector_type;
	//};
	/** 
	 * @brief Parent class for dense matrices
	 */
	template<class __value_type, class __mem_layout, class __memory_space_type, class __index_type = unsigned int >
	class dense_matrix 
	:        public matrix<__value_type, __index_type>{
	  public:
		  //typedef matrix_traits<__value_type,__index_type,__memory_space_type> my_matrix_traits;
		  typedef __mem_layout                       					  memory_layout; ///< Memory layout type: column_major or row_major
		  typedef __memory_space_type									  memory_space_type;
		  typedef matrix<__value_type, __index_type>					  base_type; ///< Basic matrix type
		  typedef typename base_type::value_type 						  value_type; ///< Type of matrix entries
		  typedef typename base_type::index_type 						  index_type; ///< Type of indices
		  //typedef typename my_matrix_traits::vector_type  					  vec_type; ///< Basic vector type used
		  typedef vector<value_type,memory_space_type,index_type>	  vec_type; ///< Basic vector type used
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
		  inline size_t memsize()       const { cuvAssert(m_vec); return m_vec->memsize(); }	///< Return matrix size in memory
		  inline const value_type* ptr()const { cuvAssert(m_vec); return m_vec->ptr(); }		///< Return device pointer to matrix entries
		  inline       value_type* ptr()      { cuvAssert(m_vec); return m_vec->ptr(); }		///< Return device pointer to matrix entries
		  inline const vec_type& vec()const { return *m_vec; }									///< Return reference to vector containing matrix entries
		  inline       vec_type& vec()      { return *m_vec; }									///< Return reference to vector containing matrix entries
		  inline const vec_type* vec_ptr()const { return m_vec; }								///< Return pointer to vector containing matrix entries
		  inline       vec_type* vec_ptr()      { return m_vec; }								///< Return pointer to vector containing matrix entries
		  inline 	   void set(const index_type& i, const index_type& j, const value_type& val);///< Set entry at position (i,j)

			virtual ~dense_matrix(){} ///< Destructor
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
			 * @brief Constructor for dense matrices that creates a matrix from a given host pointer
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

			dense_matrix(const index_type& h, const index_type& w, vec_type* p)
				:	base_type(h,w),m_vec(p)
			{
			}

		    template<class V, class I>
		  	dense_matrix(const matrix<V,I>* m)
		  	:  base_type(m->h(),m->w())
		  	{ 
		  	}
			void dealloc() ///< Deallocate matrix entries. This calls deallocation of the vector storing entries.
			{
				if(m_vec)
					delete m_vec;
				m_vec = NULL;
			}

			void alloc() ///< Allocate matrix entries: Create vector to store entries.
			{
				cuvAssert(!m_vec);
				m_vec = new vec_type(m_width * m_height);
			}

			/** 
			 * @brief Assignment operator. Assigns vector belonging to source to destination and sets source vector to NULL
			 * 
			 * @param o Source matrix
			 * 
			 * @return Matrix of same size and type of o that now owns vector of entries of o.
			 */
			  dense_matrix<value_type,memory_layout, memory_space_type,index_type>& 
			  operator=(dense_matrix<value_type,memory_layout, memory_space_type,index_type>& o){
				  if(this==&o) return *this;
				  this->dealloc();
					  //(dense_matrix<value_type,memory_layout, memory_space_type,index_type>&) (*this)  = (dense_matrix<value_type,memory_layout, memory_space_type,index_type>&) o; // copy width, height
					  //(matrix<value_type,index_type>&) (*this)  = (matrix<value_type,index_type>&) o; // copy width, height
					  (base_type&) (*this)  = (base_type&) o; // copy width, height
				  m_vec   = o.m_vec;
				  o.m_vec = NULL;                // transfer ownership of memory
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
		cout << "Device-Dense-Matrix: "<<endl;
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
