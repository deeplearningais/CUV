#ifndef __H_DENSE_MATRIX_H__
#define __H_DENSE_MATRIX_H__

#include <dense_matrix.hpp>
#include <host_vector.hpp>

namespace cuv{
	template<class __value_type, class __mem_layout=cuv::column_major, class __index_type=int>
	class host_dense_matrix 
	:        public dense_matrix<__value_type, __mem_layout, __index_type>{
	  public:
		  typedef dense_matrix<__value_type, __mem_layout, __index_type>           base_type;
		  typedef typename dense_matrix<__value_type,__mem_layout,__index_type>::memory_layout  memory_layout;
		  typedef typename matrix<__value_type,__index_type>::value_type        value_type;
		  typedef typename matrix<__value_type,__index_type>::index_type        index_type;
		  using matrix<__value_type, __index_type>::m_width;
		  using matrix<__value_type, __index_type>::m_height;
		  using base_type::m_vec;
	  private:
		  inline const value_type& operator()(const index_type& i, const index_type& j, const column_major& x) const;
		  inline const value_type& operator()(const index_type& i, const index_type& j, const row_major& x)    const;
		  inline       value_type& operator()(const index_type& i, const index_type& j, const column_major& x) ;
		  inline       value_type& operator()(const index_type& i, const index_type& j, const row_major& x)    ;
	  public:
			// life cycle
			host_dense_matrix(const index_type& h, const index_type& w) : base_type(h,w) {}
			host_dense_matrix(const index_type& h, const index_type& w, value_type* p, const bool& is_view) : base_type(h,w,p,is_view) {}
			host_dense_matrix<value_type,memory_layout,index_type>& 
			operator=(host_dense_matrix<value_type,memory_layout,index_type>& o){
				if(this==&o) return *this;
				(dense_matrix<value_type,memory_layout,index_type>&) (*this)  = (dense_matrix<value_type,memory_layout,index_type>&) o; // copy width, height
				return *this;
			}
			virtual void alloc();

			// element access
		  inline const value_type& operator()(const index_type& i, const index_type& j) const;
		  inline       value_type& operator()(const index_type& i, const index_type& j);
	};

	/*
	 * memory allocation
	 *
	 */
	template<class V, class M, class I>
	void
	host_dense_matrix<V,M,I>::alloc() { m_ptr = new host_vector(this->n()); }


	/*
	 * element access for dense host matrix
	 *
	 */
	template<class V, class M, class I>
	const typename matrix<V,I>::value_type&
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, const column_major& x) const{ return (*m_vec)[ this->h()*j + i]; }

	template<class V, class M, class I>
	const typename matrix<V,I>::value_type&
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, const row_major& x)    const{ return (*m_vec)[ this->w()*i + j]; }

	template<class V, class M, class I>
	const typename matrix<V,I>::value_type&
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j)    const{ return (*this)(i,j,memory_layout()); }

	template<class V, class M, class I>
	typename matrix<V,I>::value_type&
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, const column_major& x) { return (*m_vec)[ this->h()*j + i]; }

	template<class V, class M, class I>
	typename matrix<V,I>::value_type&
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, const row_major& x)    { return (*m_vec)[ this->w()*i + j]; }

	template<class V, class M, class I>
	typename matrix<V,I>::value_type&
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j)    { return (*this)(i,j,memory_layout()); }

}

#endif /* __MATRIX_H__ */
