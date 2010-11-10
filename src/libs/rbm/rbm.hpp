#ifndef __RBM__HPP__
#define __RBM__HPP__


namespace cuv{
namespace libs{
namespace rbm{
	/**
	 * set a matrix to consecutive binary numbers in the columns, starting with the number `start'
	 *
	 * @param m      the target matrix
	 * @param start  the value of the first column
	 */
	template<class __matrix_type>
	void set_binary_sequence(__matrix_type& m, const int& start);

	/**
	 * apply sigmoid column-wise with the temperature specified for each column
	 *
	 * @param m    source and target matrix
	 * @param temp the temperature (one value per column)
	 */
	template<class __matrix_type,class __vector_type>
	void sigm_temperature(__matrix_type& m, const __vector_type& temp);
}
}
}

#endif /* __RBM__HPP__ */
