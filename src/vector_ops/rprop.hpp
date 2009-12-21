#ifndef __RPROP_HPP__
#define __RPROP_HPP__

#include <cuv_general.hpp>
#include "dev_vector.hpp"
#include "host_vector.hpp"

namespace cuv{


	template<class __vector_type, class __old_vector_type>
	void rprop(__vector_type& W, __vector_type& dW, __old_vector_type& dW_old, __vector_type& rate);

	template<class __vector_type>
	void learn_step_weight_decay(__vector_type& W, __vector_type& dW, const float& learnrate, const float& decay);

}


#endif /* __RPROP_HPP__ */
