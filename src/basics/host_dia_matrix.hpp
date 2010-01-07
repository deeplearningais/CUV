#ifndef __HOST_SPARSE_MATRIX_HPP__
#define __HOST_SPARSE_MATRIX_HPP__

#include "host_vector.hpp"
#include "sparse_matrix.hpp"

namespace cuv{
	template<class __value_type, class __index_type=unsigned int>
	class host_dia_matrix
	:	public dia_matrix<__value_type, __index_type, host_vector<__value_type, __index_type>, host_vector<int, unsigned int> >{
		public:
		  typedef host_vector<__value_type, __index_type>          vec_type;
		  typedef host_vector<int, unsigned int>                   intvec_type;
		  typedef dia_matrix<__value_type, __index_type, vec_type, intvec_type> base_type;
		  typedef typename base_type::value_type                   value_type;
		  typedef typename base_type::index_type                   index_type;
		protected:
		public:
		  ~host_dia_matrix(){}
		  host_dia_matrix(const index_type& h, const index_type& w, const int& num_dia, const int& stride)
			  :base_type(h,w,num_dia,stride)
		  {
		  }
	};
};

#endif
