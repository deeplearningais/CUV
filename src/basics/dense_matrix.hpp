#ifndef __DENSE_MATRIX_HPP__
#define __DENSE_MATRIX_HPP__
#include <matrix.hpp>

namespace cuv{
  struct memory_layout_tag{};
	struct column_major : public memory_layout_tag{};
	struct row_major    : public memory_layout_tag{};

	template<class __value_type, class __mem_layout, class __index_type>
	class dense_matrix 
	:        public matrix<__value_type, __index_type>{
	  public:
		  typedef __mem_layout memory_layout;
		private:
		  value_type* m_ptr;
			bool        m_is_view;
		public:
			inline size_t memsize()const{ return n()*sizeof(value_type); }
		  inline const value_type* ptr()const { return m_ptr; }
		  inline       value_type* ptr()      { return m_ptr; }
			dense_matrix(const index_type& h, const index_type& w)
			  : matrix(h,w), m_ptr(NULL), m_is_view(false) {}
			dense_matrix(const index_type& h, const index_type& w, value_type* p, const bool& is_view)
			  : matrix(h,w), m_ptr(p), m_is_view(is_view) {}
			inline host_dense_matrix<value_type,memory_layout, index_type>& operator=(host_dense_matrix<value_type, memory_layout, index_type>& o);
	}

	/*
	 * assignment operator
	 *
	 */
	template<class V, class M, class I>
	dense_matrix<value_type,memory_layout, index_type>& 
	dense_matrix<V,M,I>::operator=(dense_matrix<value_type, memory_layout, index_type>& o){
	  m_width  = o.w();
	  m_height = o.h();
		m_ptr    = o.ptr();
		m_is_view = o.m_is_view;
	  o.m_ptr = NULL;                // transfer ownership of memory
	}
}

#endif /* __DENSE_MATRIX_HPP__ */
