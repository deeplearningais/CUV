#ifndef __DEV_VECTOR_HPP__
#define __DEV_VECTOR_HPP__

#include "vector.hpp"

namespace cuv{

template<class __value_type, class __index_type>
class dev_vector
:    public vector<__value_type, __index_type>
{
  public:
	  typedef __value_type value_type;
		typedef __index_type index_type;
		typedef vector<__value_type, __index_type> base_type;
		using base_type::m_ptr;
	public:
		dev_vector(size_t s)
		:   base_type(s) { }
		dev_vector(size_t s, value_type* p, bool is_view)
		:   base_type(s,p,is_view) { }
		virtual void alloc();
		virtual void dealloc();
};

} // cuv

#endif

