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
	struct column_major : public memory_layout_tag{};
	struct row_major    : public memory_layout_tag{};

	template<class __value_type, class __mem_layout, class __index_type>
	class dense_matrix 
	:        public matrix<__value_type, __index_type>{
	  public:
		  typedef __mem_layout                       memory_layout;
		  typedef matrix<__value_type, __index_type> base_type;
		  typedef typename matrix<__value_type, __index_type>::value_type value_type;
		  typedef typename matrix<__value_type, __index_type>::index_type index_type;
		protected:
		public:
			~dense_matrix(){}
			dense_matrix(const index_type& h, const index_type& w)
				: base_type(h,w) {}
			dense_matrix(const index_type& h, const index_type& w, value_type* p, bool is_view = false) { }
	};
}

#endif /* __DENSE_MATRIX_HPP__ */
