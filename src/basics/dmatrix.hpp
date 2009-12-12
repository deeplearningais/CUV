#ifndef __H_DENSE_MATRIX_H__
#define __H_DENSE_MATRIX_H__

#include <dense_matrix.hpp>
namespace cuv{
	template<class __value_type, class __mem_layout, class __index_type>
	class host_dense_matrix 
	:        public dense_matrix<__value_type, __index_type, __mem_layout>{
	  private:
		  inline const value_type& operator()(const index_type& i, const index_type& j, const column_major& x) const;
		  inline const value_type& operator()(const index_type& i, const index_type& j, const row_major& x)    const;
		  inline       value_type& operator()(const index_type& i, const index_type& j, const column_major& x) ;
		  inline       value_type& operator()(const index_type& i, const index_type& j, const row_major& x)    ;
	  public:
		  inline const value_type& operator()(const index_type& i, const index_type& j) const;
		  inline       value_type& operator()(const index_type& i, const index_type& j);
	};

	template<class __value_type, class __mem_layout, class __index_type>
	class dev_dense_matrix
	:        public dense_matrix<__value_type, __index_type>{
	  public:
	};




	/*
	 * element access for dense host matrix
	 *
	 */
	template<class V, class M, class I>
	const value_type&
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, const column_major& x) const{ return m_ptr[ h()*j + i]; }

	template<class V, class M, class I>
	const value_type&
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, const row_major& x)    const{ return m_ptr[ w()*i + j]; }

	template<class V, class M, class I>
	const value_type&
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j)    const{ creturn (*this)(i,j,memory_layout()); }

	template<class V, class M, class I>
	value_type&
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, const column_major& x) { return m_ptr[ h()*j + i]; }

	template<class V, class M, class I>
	value_type&
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j, const row_major& x)    { return m_ptr[ w()*i + j]; }

	template<class V, class M, class I>
	value_type&
	host_dense_matrix<V,M,I>::operator()(const index_type& i, const index_type& j)    { creturn (*this)(i,j,memory_layout()); }

}

#endif /* __MATRIX_H__ */
