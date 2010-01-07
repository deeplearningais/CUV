#ifndef __DEV_SPARSE_MATRIX_HPP__
#define __DEV_SPARSE_MATRIX_HPP__

#include "dev_vector.hpp"
#include "sparse_matrix.hpp"

namespace cuv{
	template<class __value_type, class __index_type=unsigned int>
	class dev_dia_matrix
	:	public dia_matrix<__value_type, __index_type, dev_vector<__value_type, __index_type>, dev_vector<int> >{
		public:
		  typedef dev_vector<__value_type, __index_type>          vec_type;
		  typedef dev_vector<int>                                 intvec_type;
		  typedef dia_matrix<__value_type, __index_type, vec_type, intvec_type> base_type;
		  typedef typename base_type::value_type                   value_type;
		  typedef typename base_type::index_type                   index_type;
		protected:
		public:
		  ~dev_dia_matrix(){}
		  dev_dia_matrix(const index_type& h, const index_type& w, const int& num_dia, const int& stride)
			  :base_type(h,w,num_dia,stride)
		  {
		  }
	};
};

#endif

