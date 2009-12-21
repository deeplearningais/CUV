#ifndef __RPROP_HPP__
#define __RPROP_HPP__

#include <cuv_general.hpp>
#include "dev_vector.hpp"
#include "host_vector.hpp"

namespace cuv{


	template<class __vector_type, class __old_vector_type>
	void rprop(__vector_type& dW, __old_vector_type& dW_old, __vector_type& rate);

}


#endif /* __RPROP_HPP__ */
