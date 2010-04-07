/** 
 * @file dev_dense_matrix.hpp
 * @brief dense matrix on device
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2010-03-21
 */
#ifndef __DEV_DENSE_MATRIX_H__
#define __DEV_DENSE_MATRIX_H__
#include <tools/cuv_general.hpp>
#include <basics/dense_matrix.hpp>
#include <basics/dev_vector.hpp>

namespace cuv{
	template<class __value_type, class __mem_layout=cuv::column_major, class __index_type=unsigned int>
		/** 
		 * @brief Class for dense device(=GPU memory) matrices.
		 */
	class dev_dense_matrix
	:        public dense_matrix<__value_type, __mem_layout, dev_vector<__value_type,__index_type>,__index_type>{
		public:
		  typedef __mem_layout        memory_layout;
		  typedef dev_vector<__value_type, __index_type>                        vec_type;	///< Basic vector type used
		  typedef dense_matrix<__value_type, __mem_layout,vec_type, __index_type>        base_type;	///< Basic dense matrix type
		  typedef typename base_type::value_type value_type;
		  typedef typename base_type::index_type index_type;
		  using base_type::m_width;
		  using base_type::m_height;
		  using base_type::m_vec;
		//protected:
		  //vec_type* m_vec; ///< Pointer to vector containing matrix entries
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

		  /*
		   * Construction
		   */

			/** 
			 * @brief Creates matrix of same type and size as given matrix
			 * 
			 * @param m Matrix of type and size used for returned matrix
			 *
			 * Allocates memory for entries in main memory.
			 *
			 * @return Matrix of size m.h() x m.w() and same type as matrix m.
			 */
			  template<class V, class I>
			  dev_dense_matrix(const matrix<V,I>* m)
			  :  base_type(m->h(),m->w())
			  { 
				  //this->alloc(); 
			  }

			/** 
			 * @brief Creates matrix of weight w and height h.
			 * 
			 * @param h Height of matrix
			 * @param w Width of matrix
			 */
			  dev_dense_matrix(const index_type& h, const index_type& w)
				:  base_type(h,w){}

			/** 
			 * @brief Creates matrix as view on given host vector.
			 * 
			 * @param h Heights of matrix
			 * @param w Width of matrix
			 * @param p Pointer to vector containing matrix entries.
			 *
			 * This function does not allocate any memory but uses the memory belonging to p.
			 * No responsibility for the memory that p points to is taken, i.e. when returned matrix is destoyed, the vector p is not destroyed.
			 *
			 */
			  dev_dense_matrix(const index_type& h, const index_type& w, dev_vector<value_type,index_type>* p, bool is_view = false)
				:  base_type(h,w,p->ptr(),is_view) { } // do not alloc!
			/** 
			 * @brief Creates matrix from given host vector.
			 * 
			 * @param h Heights of matrix
			 * @param w Width of matrix
			 * @param p  Pointer to vector containing matrix entries.
			 * @param is_view If true will not take responsibility of memory at p. Otherwise will dealloc p on destruction.
			 */
			  dev_dense_matrix(const index_type& h, const index_type& w, value_type* p, bool is_view = false)
				:	base_type(h,w,p,is_view) {}
			  //~dev_dense_matrix(){ dealloc(); } ///< Destructor: Deallocate matrix memory if is_view is false

			/** 
			 * @brief Assignment operator. Assigns vector belonging to source to destination and sets source vector to NULL
			 * 
			 * @param o Source matrix
			 * 
			 * @return Matrix of same size and type of o that now owns vector of entries of o.
			 */
			  dev_dense_matrix<value_type,memory_layout,index_type>& 
			  operator=(dev_dense_matrix<value_type,memory_layout,index_type>& o){
				  if(this==&o) return *this;
				  this->dealloc();
					  (dense_matrix<value_type,memory_layout,vec_type,index_type>&) (*this)  = (dense_matrix<value_type,memory_layout,vec_type,index_type>&) o; // copy width, height
				  m_vec   = o.m_vec;
				  o.m_vec = NULL;                // transfer ownership of memory
				  return *this;
		  }

		  /*
		   * Memory management
		   */
			  /** 
			   * @brief Allocate memory for matrix entries 
			   */
		  //void alloc() 
		  //{   
			  //cuvAssert(!m_vec);
			  //m_vec = new dev_vector<value_type,index_type>(this->n());
		  //}
		  /** 
		   * @brief Deallocate memory for matrix entries. Does nothing if is_view is true.
		   */
		  //void dealloc() 
		  //{
			  //if(m_vec)
				  //delete m_vec;
			  //m_vec = NULL;
		  //};
	};

	/*
	 * element access for dense dev matrix
	 *
	 */
	template<class V, class M, class I>
	const typename dev_dense_matrix<V,M,I>::value_type
	dev_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, const column_major& x) const{ return (*m_vec)[ this->h()*j + i]; }

	template<class V, class M, class I>
	const typename dev_dense_matrix<V,M,I>::value_type
	dev_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, const row_major& x)    const{ return (*m_vec)[ this->w()*i + j]; }

	template<class V, class M, class I>
	const typename dev_dense_matrix<V,M,I>::value_type
	dev_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j)    const{ return (*this)(i,j,memory_layout()); }

	template<class V, class M, class I>
	typename dev_dense_matrix<V,M,I>::value_type
	dev_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, const column_major& x) { return (*m_vec)[ this->h()*j + i]; }

	template<class V, class M, class I>
	typename dev_dense_matrix<V,M,I>::value_type
	dev_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, const row_major& x)    { return (*m_vec)[ this->w()*i + j]; }

	template<class V, class M, class I>
	typename dev_dense_matrix<V,M,I>::value_type
	dev_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j)    { return (*this)(i,j,memory_layout()); }

	/** 
	 * @brief Matrix traits for memory type (host/device) and memory layout (row major/column major)
	 */
	template<class V, class M, class I>
		struct matrix_traits<dev_dense_matrix<V,M,I> >{
			typedef dev_memory_space memory_space_type; ///< Trait for memory type (host/device)
			typedef M                memory_layout_type; ///< Trait for memory layout (row major / column major)
		};


	/*
	 * Change values in dense dev matrix
	 *
	 */
	template<class V, class M, class I>
	void
	dev_dense_matrix<V,M,I>::set(const index_type& i, const index_type& j, const value_type& val, const column_major&) { m_vec->set( this->h()*j + i, val); };

	template<class V, class M, class I>
	void
	dev_dense_matrix<V,M,I>::set(const index_type& i, const index_type& j, const value_type& val, const row_major&) { m_vec->set( this->w()*i + j, val); };

	template<class V, class M, class I>
	void
	dev_dense_matrix<V,M,I>::set(const index_type& i, const index_type& j, const value_type& val) { this->set(i, j, val, memory_layout()); };

}

#include <iostream>
namespace std{
	template<class T, class M, class I>
	/** 
	 * @brief Return stream containing matrix entries for debugging
	 * 
	 * @param o Output stream
	 * @param w2 Matrix to output
	 */
	ostream& 
	operator<<(ostream& o, const cuv::dev_dense_matrix<T,M,I>& w2){
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

#endif

