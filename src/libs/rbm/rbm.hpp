#ifndef __RBM__HPP__
#define __RBM__HPP__


namespace cuv{
namespace libs{
namespace rbm{
	template<class __matrix_type>
	void set_binary_sequence(__matrix_type& m, const int& start);
	template<class __matrix_type,class __vector_type>
	void sigm_temperature(__matrix_type& m, const __vector_type& temp);
}
}
}

#endif /* __RBM__HPP__ */
