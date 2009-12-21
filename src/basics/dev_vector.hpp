#ifndef __DEV_VECTOR_HPP__
#define __DEV_VECTOR_HPP__

#include <cuv_general.hpp>
#include "vector.hpp"

namespace cuv{

template<class __value_type, class __index_type=unsigned int>
class dev_vector
:    public vector<__value_type, __index_type>
{
  public:
	  typedef __value_type                       value_type;
	  typedef __index_type                       index_type;
	  typedef vector<__value_type, __index_type> base_type;
	  typedef dev_memory_space                   memspace_type;
	  using base_type::m_ptr;
	public:
		dev_vector(size_t s)
		:   base_type(s) { alloc(); }
		dev_vector(size_t s, value_type* p, bool is_view)
		:   base_type(s,p,is_view) { alloc(); }
		value_type operator[](size_t t);
		virtual void alloc();
		virtual void dealloc();
};

template<class V, class I>
struct vector_traits<dev_vector<V,I> >{
	typedef dev_memory_space memory_space_type;
};

} // cuv

#endif

