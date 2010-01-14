#ifndef __H_DENSE_MATRIX_H__
#define __H_DENSE_MATRIX_H__

#include <cuv_general.hpp>
#include <dense_matrix.hpp>
#include <host_vector.hpp>

namespace cuv{
	template<class __value_type, class __mem_layout=cuv::column_major, class __index_type=unsigned int>
	class host_dense_matrix 
	:        public dense_matrix<__value_type, __mem_layout, __index_type>{
		public:
			typedef dense_matrix<__value_type, __mem_layout, __index_type>         base_type;
			typedef host_vector<__value_type, __index_type>                        vec_type;
			typedef typename dense_matrix<__value_type,__mem_layout,__index_type>::memory_layout  memory_layout;
			typedef typename matrix<__value_type,__index_type>::value_type        value_type;
			typedef typename matrix<__value_type,__index_type>::index_type        index_type;
			using matrix<__value_type, __index_type>::m_width;
			using matrix<__value_type, __index_type>::m_height;
		protected:
			vec_type* m_vec;
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
			inline const value_type operator()(const index_type& i, const index_type& j) const; 
			inline       value_type operator()(const index_type& i, const index_type& j);
			inline size_t memsize()       const { cuvAssert(m_vec); return m_vec->memsize(); }
			inline const value_type* ptr()const { cuvAssert(m_vec); return m_vec->ptr(); }
			inline       value_type* ptr()      { cuvAssert(m_vec); return m_vec->ptr(); }
			inline const vec_type& vec()  const { return *m_vec; }
			inline       vec_type& vec()        { return *m_vec; }
			inline const vec_type* vec_ptr()  const { return m_vec; }
			inline       vec_type* vec_ptr()        { return m_vec; }
			inline 		 void set(const index_type& i, const index_type& j, const value_type& val);

			/*
			 * Life cycle
			 */
			template<class V, class I>
				host_dense_matrix(const matrix<V,I>* m)
				:  base_type(m->h(),m->w()), m_vec(NULL)
				{ 
					this->alloc(); 
				}
			host_dense_matrix(const index_type& h, const index_type& w) 
				:	base_type(h,w), m_vec(NULL){ alloc(); }
			host_dense_matrix(const index_type& h, const index_type& w, host_vector<value_type,index_type>* p) 
				:	base_type(h,w), m_vec(p) {} // do not alloc!
			host_dense_matrix(const index_type& h, const index_type& w, value_type* p, bool is_view)
				:	base_type(h,w) { m_vec = new vec_type(h*w,p,is_view); }
			~host_dense_matrix(){ dealloc(); }
			host_dense_matrix<value_type,memory_layout,index_type>& 
				operator=(host_dense_matrix<value_type,memory_layout,index_type>& o){
					if(this==&o) return *this;
					this->dealloc();
					(dense_matrix<value_type,memory_layout,index_type>&) (*this)  = (dense_matrix<value_type,memory_layout,index_type>&) o; // copy width, height
					m_vec   = o.m_vec;
					o.m_vec = NULL;                // transfer ownership of memory
					return *this;
				}

			/*
			 * Memory Management
			 */
			void alloc();
			void dealloc();

	};

	/*
	 * memory allocation
	 *
	 */
	template<class V, class M, class I>
	void
	host_dense_matrix<V,M,I>::alloc() { 
		cuvAssert(!m_vec);
		m_vec = new host_vector<value_type,index_type>(this->n()); 
	}

	template<class V, class M, class I>
	void
	host_dense_matrix<V,M,I>::dealloc() {
		if(m_vec)
			delete m_vec;
		m_vec = NULL;
	}


	/*
	 * element access for dense host matrix
	 *
	 */
	template<class V, class M, class I>
	const typename matrix<V,I>::value_type
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, column_major) const{ return (*m_vec)[ this->h()*j + i]; }

	template<class V, class M, class I>
	const typename matrix<V,I>::value_type
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, row_major)    const{ return (*m_vec)[ this->w()*i + j]; }

	template<class V, class M, class I>
	typename matrix<V,I>::value_type
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, column_major) { return (*m_vec)[ this->h()*j + i]; }

	template<class V, class M, class I>
	typename matrix<V,I>::value_type
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, row_major)    { return (*m_vec)[ this->w()*i + j]; }

	template<class V, class M, class I>
	typename matrix<V,I>::value_type
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j)      { return (*this)(i,j,memory_layout()); }

	template<class V, class M, class I>
	const typename matrix<V,I>::value_type
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j) const{ return (*this)(i,j,memory_layout()); }

	template<class V, class M, class I>
		struct matrix_traits<host_dense_matrix<V,M,I> >{
			typedef host_memory_space memory_space_type;
			typedef M                 memory_layout_type;
		};

	/*
	 * Change values in dense host matrix
	 *
	 */
	template<class V, class M, class I>
	void
	host_dense_matrix<V,M,I>::set(const index_type& i, const index_type& j, const value_type& val, column_major) { (*m_vec)[ this->h()*j + i] = val; };

	template<class V, class M, class I>
	void
	host_dense_matrix<V,M,I>::set(const index_type& i, const index_type& j, const value_type& val, row_major) { (*m_vec)[ this->w()*i + j] = val; };

	template<class V, class M, class I>
	void
	host_dense_matrix<V,M,I>::set(const index_type& i, const index_type& j, const value_type& val) { this->set(i, j, val, memory_layout()); };


}

#endif /* __MATRIX_H__ */
