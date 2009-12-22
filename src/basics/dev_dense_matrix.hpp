#ifndef __DEV_DENSE_MATRIX_H__
#define __DEV_DENSE_MATRIX_H__
#include <cuv_general.hpp>
#include <dense_matrix.hpp>
#include <dev_vector.hpp>

namespace cuv{
	template<class __value_type, class __mem_layout=cuv::column_major, class __index_type=unsigned int>
	class dev_dense_matrix
	:        public dense_matrix<__value_type, __mem_layout, __index_type>{
		public:
		  typedef __mem_layout        memory_layout;
		  typedef dev_vector<__value_type, __index_type>                        vec_type;
		  typedef dense_matrix<__value_type, __mem_layout, __index_type>        base_type;
		  typedef typename matrix<__value_type,__index_type>::value_type        value_type;
		  typedef typename matrix<__value_type,__index_type>::index_type        index_type;
		  using matrix<__value_type, __index_type>::m_width;
		  using matrix<__value_type, __index_type>::m_height;
		protected:
		  vec_type* m_vec;
		public:
		  inline size_t memsize()       const { cuvAssert(m_vec); return m_vec->memsize(); }
		  inline const value_type* ptr()const { cuvAssert(m_vec); return m_vec->ptr(); }
		  inline       value_type* ptr()      { cuvAssert(m_vec); return m_vec->ptr(); }
		  inline const vec_type& vec()const { return *m_vec; }
		  inline       vec_type& vec()      { return *m_vec; }
		  inline const vec_type* vec_ptr()const { return m_vec; }
		  inline       vec_type* vec_ptr()      { return m_vec; }
		  template<class V, class I>
		  dev_dense_matrix(const matrix<V,I>* m)
		  :  base_type(m->h(),m->w()), m_vec(NULL)
		  { 
			  this->alloc(); 
		  }

		  dev_dense_matrix(const index_type& h, const index_type& w)
			:  base_type(h,w), m_vec(NULL) { alloc(); }

		  dev_dense_matrix(const index_type& h, const index_type& w, dev_vector<value_type,index_type>* p)
			:  base_type(h,w), m_vec(p) { } // do not alloc!

		  dev_dense_matrix<value_type,memory_layout,index_type>& 
		  operator=(dev_dense_matrix<value_type,memory_layout,index_type>& o){
			  if(this==&o) return *this;
			  this->dealloc();
			  (dense_matrix<value_type,memory_layout,index_type>&) (*this)  = (dense_matrix<value_type,memory_layout,index_type>&) o; // copy width, height
			  m_vec   = o.m_vec;
			  o.m_vec = NULL;                // transfer ownership of memory
			  return *this;
		  }
		  void alloc(){   
			  cuvAssert(!m_vec);
			  m_vec = new dev_vector<value_type,index_type>(this->n());
		  }
		  void dealloc(){
			  if(m_vec)
				  delete m_vec;
			  m_vec = NULL;
		  };
		  // element access
		  inline const value_type operator()(const index_type& i, const index_type& j) const;
		  inline       value_type operator()(const index_type& i, const index_type& j);
	  private:
		  inline const value_type operator()(const index_type& i, const index_type& j, const column_major& x) const;
		  inline const value_type operator()(const index_type& i, const index_type& j, const row_major& x)    const;
		  inline       value_type operator()(const index_type& i, const index_type& j, const column_major& x) ;
		  inline       value_type operator()(const index_type& i, const index_type& j, const row_major& x)    ;
	};

	/*
	 * element access for dense dev matrix
	 *
	 */
	template<class V, class M, class I>
	const typename matrix<V,I>::value_type
	dev_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, const column_major& x) const{ return (*m_vec)[ this->h()*j + i]; }

	template<class V, class M, class I>
	const typename matrix<V,I>::value_type
	dev_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, const row_major& x)    const{ return (*m_vec)[ this->w()*i + j]; }

	template<class V, class M, class I>
	const typename matrix<V,I>::value_type
	dev_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j)    const{ return (*this)(i,j,memory_layout()); }

	template<class V, class M, class I>
	typename matrix<V,I>::value_type
	dev_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, const column_major& x) { return (*m_vec)[ this->h()*j + i]; }

	template<class V, class M, class I>
	typename matrix<V,I>::value_type
	dev_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, const row_major& x)    { return (*m_vec)[ this->w()*i + j]; }

	template<class V, class M, class I>
	typename matrix<V,I>::value_type
	dev_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j)    { return (*this)(i,j,memory_layout()); }

	template<class V, class M, class I>
		struct matrix_traits<dev_dense_matrix<V,M,I> >{
			typedef dev_memory_space memory_space_type;
			typedef M                memory_layout_type;
		};
}

#endif

