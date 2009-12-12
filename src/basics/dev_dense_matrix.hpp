#ifndef __D_DENSE_MATRIX_H__
#define __D_DENSE_MATRIX_H__
#include <dense_matrix.hpp>

namespace cuv{
	template<class __value_type, class __mem_layout, class __index_type>
	class dev_dense_matrix
	:        public dense_matrix<__value_type, __mem_layout, __index_type>{
		public:
		  typedef dense_matrix<__value_type, __mem_layout, __index_type>        base_type;
		  using typename matrix<__value_type,__index_type>::value_type;
		  using typename matrix<__value_type,__index_type>::index_type;
		  using base_type::m_ptr;
		public:
		  dev_dense_matrix(const index_type& h, const index_type& w);
		  dev_dense_matrix(const index_type& h, const index_type& w, value_type* p, const bool& is_view);
		  ~dev_dense_matrix();
		  void alloc();
		  void dealloc();
	};
}

#endif

