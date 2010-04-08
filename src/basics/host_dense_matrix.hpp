/** 
 * @file host_dense_matrix.hpp
 * @brief dense matrix on host
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2010-03-21
 */
#ifndef __H_DENSE_MATRIX_H__
#define __H_DENSE_MATRIX_H__

#include <tools/cuv_general.hpp>
#include "dense_matrix.hpp"
#include "host_vector.hpp"

namespace cuv{
	template<class __value_type, class __mem_layout=cuv::column_major, class __index_type=unsigned int>
		/** 
		 * @brief Class for dense host(=CPU memory) matrices.
		 */
	class host_dense_matrix 
	:        public dense_matrix<__value_type, __mem_layout, host_vector<__value_type, __index_type>, __index_type>{
		public:
		  	typedef __mem_layout        memory_layout;
			typedef host_vector<__value_type, __index_type>                        vec_type; ///< Basic vector type used
			typedef dense_matrix<__value_type, __mem_layout, vec_type, __index_type>         base_type; ///< Basic dense matrix type
			typedef typename base_type::value_type value_type;
			typedef typename base_type::index_type index_type;
			using base_type::m_width;
			using base_type::m_height;
			using base_type::m_vec;
		//protected:
			//vec_type* m_vec; ///< Pointer to vector containing matrix entries
		private:
			inline const value_type operator()(const index_type& i, const index_type& j, column_major) const;
			inline const value_type operator()(const index_type& i, const index_type& j, row_major)    const;
			inline       value_type operator()(const index_type& i, const index_type& j, column_major) ;
			inline       value_type operator()(const index_type& i, const index_type& j, row_major)    ;
			inline		 void set(const index_type& i, const index_type& j, const value_type& val, column_major);
			inline		 void set(const index_type& i, const index_type& j, const value_type& val, row_major);
		public:
			/*
			 * Member access
			 */
			// do _not_ return a reference, we want to be compatible with device memory classes and there references do not work
			inline const value_type operator()(const index_type& i, const index_type& j) const; ///< Read entry at position (i,j)
			inline       value_type operator()(const index_type& i, const index_type& j); ///< Read entry at position (i,j)
			inline size_t memsize()       const { cuvAssert(m_vec); return m_vec->memsize(); } ///< Return matrix size in memory
			inline const value_type* ptr()const { cuvAssert(m_vec); return m_vec->ptr(); } ///< Return pointer to matrix entries
			inline       value_type* ptr()      { cuvAssert(m_vec); return m_vec->ptr(); } ///< Return pointer to matrix entries
			inline const vec_type& vec()  const { return *m_vec; } ///< Return reference to vector containing matrix entries
			inline       vec_type& vec()        { return *m_vec; } ///< Return reference to vector containing matrix entries
			inline const vec_type* vec_ptr()  const { return m_vec; } ///< Return pointer to vector containing matrix entries
			inline       vec_type* vec_ptr()        { return m_vec; } ///< Return pointer to vector containing matrix entries
			inline 		 void set(const index_type& i, const index_type& j, const value_type& val); ///< Set entry at position (i,j)

			/*
			 * Life cycle
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
				host_dense_matrix(const matrix<V,I>* m)
				:  base_type(m->h(),m->w())
				{ 
				}
			/** 
			 * @brief Creates matrix of weight w and height h.
			 * 
			 * @param h Height of matrix
			 * @param w Width of matrix
			 */
			host_dense_matrix(const index_type& h, const index_type& w) 
				:	base_type(h,w){}
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
			host_dense_matrix(const index_type& h, const index_type& w, host_vector<value_type,index_type>* p, bool is_view = false) 
				:	base_type(h,w,p->ptr()) {} // do not alloc!
			/** 
			 * @brief Creates matrix from given host vector.
			 * 
			 * @param h Heights of matrix
			 * @param w Width of matrix
			 * @param p  Pointer to vector containing matrix entries.
			 * @param is_view If true will not take responsibility of memory at p. Otherwise will dealloc p on destruction.
			 */
			host_dense_matrix(const index_type& h, const index_type& w, value_type* p, bool is_view = false)
				:	base_type(h,w,p,is_view) { }
			//~host_dense_matrix(){ dealloc(); } ///< Destructor: Deallocate matrix memory if is_view is false
			/** 
			 * @brief Assignment operator. Assigns vector belonging to source to destination and sets source vector to NULL
			 * 
			 * @param o Source matrix
			 * 
			 * @return Matrix of same size and type of o that now owns vector of entries of o.
			 */
			host_dense_matrix<value_type,memory_layout,index_type>& 
				operator=(host_dense_matrix<value_type,memory_layout,index_type>& o){
					if(this==&o) return *this;
					this->dealloc();
					(dense_matrix<value_type,memory_layout,vec_type,index_type>&) (*this)  = (dense_matrix<value_type,memory_layout,vec_type,index_type>&) o; // copy width, height
					m_vec   = o.m_vec;
					o.m_vec = NULL;                // transfer ownership of memory
					return *this;
				}

			/*
			 * Memory Management
			 */
			//void alloc(); ///< Allocate memory for matrix entries
			//void dealloc(); ///< Deallocate memory for matrix entries. Does nothing if is_view is true.

	};

	/*
	 * memory allocation
	 *
	 */
	//template<class V, class M, class I>
	//void
	//host_dense_matrix<V,M,I>::alloc() { 
		//m_vec = new host_vector<value_type,index_type>(this->n()); 
	//}

	//template<class V, class M, class I>
	//void
	//host_dense_matrix<V,M,I>::dealloc() {
		//if(m_vec)
			//delete m_vec;
		//m_vec = NULL;
	//}


	/*
	 * element access for dense host matrix
	 *
	 */
	template<class V, class M, class I>
	const typename host_dense_matrix<V,M,I>::value_type
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, column_major) const{ return (*m_vec)[ this->h()*j + i]; }

	template<class V, class M, class I>
	const typename host_dense_matrix<V,M,I>::value_type
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, row_major)    const{ return (*m_vec)[ this->w()*i + j]; }

	template<class V, class M, class I>
	typename host_dense_matrix<V,M,I>::value_type
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, column_major) { return (*m_vec)[ this->h()*j + i]; }

	template<class V, class M, class I>
	typename host_dense_matrix<V,M,I>::value_type
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, row_major)    { return (*m_vec)[ this->w()*i + j]; }

	template<class V, class M, class I>
	typename host_dense_matrix<V,M,I>::value_type
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j)      { return (*this)(i,j,memory_layout()); }

	template<class V, class M, class I>
	const typename host_dense_matrix<V,M,I>::value_type
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j) const{ return (*this)(i,j,memory_layout()); }

	/** 
	 * @brief Matrix traits for memory type (host/device) and memory layout (row major/column major)
	 */
	//template<class V, class M, class I>
		//struct matrix_traits<host_dense_matrix<V,M,I> >{
			//typedef host_memory_space memory_space_type; ///< Trait for memory type (host/device)
			//typedef M                 memory_layout_type; ///< Trait for memory layout (row major / column major)
		//};

	/*
	 * Change values in dense host matrix
	 *
	 */
	template<class V, class M, class I>
	void
	host_dense_matrix<V,M,I>::set(const index_type& i, const index_type& j, const value_type& val, column_major) { m_vec->set( this->h()*j + i, val); };

	template<class V, class M, class I>
	void
	host_dense_matrix<V,M,I>::set(const index_type& i, const index_type& j, const value_type& val, row_major) { m_vec->set( this->w()*i + j, val); };

	template<class V, class M, class I>
	void
	host_dense_matrix<V,M,I>::set(const index_type& i, const index_type& j, const value_type& val) { this->set(i, j, val, memory_layout()); };


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
	operator<<(ostream& o, const cuv::host_dense_matrix<T,M,I>& w2){
		cout << "Host-Dense-Matrix: "<<endl;
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

#endif /* __MATRIX_H__ */
