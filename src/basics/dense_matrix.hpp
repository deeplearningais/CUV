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
		  typedef __mem_layout                       memory_layout;
		  typedef matrix<__value_type, __index_type> base_type;
		  using typename matrix<__value_type, __index_type>::value_type;
		  using typename matrix<__value_type, __index_type>::index_type;
		private:
		  value_type* m_ptr;
		  bool        m_is_view;
		public:
			inline size_t memsize()const{ return this->n()*sizeof(value_type); }
			inline const value_type* ptr()const { return m_ptr; }
			inline       value_type* ptr()      { return m_ptr; }
			dense_matrix(const index_type& h, const index_type& w)
				: base_type(h,w), m_ptr(NULL), m_is_view(false) {}
			dense_matrix(const index_type& h, const index_type& w, value_type* p, const bool& is_view)
				: base_type(h,w), m_ptr(p), m_is_view(is_view) {}

			dense_matrix<value_type, memory_layout, index_type>& 
				operator=(dense_matrix<value_type, memory_layout, index_type>& o){
					if(this == &o)
						return *this;
					(matrix<value_type,index_type>&) (*this)  = (matrix<value_type,index_type>&) o; // copy width, height
					m_ptr       = o.ptr();
					m_is_view   = o.m_is_view;
					if(! o.m_is_view )
						o.m_ptr = NULL;                // transfer ownership of memory
					return *this;
				}
	};
}

#endif /* __DENSE_MATRIX_HPP__ */
