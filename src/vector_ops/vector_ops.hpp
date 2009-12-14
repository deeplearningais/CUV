#ifndef __VECTOR_OPS_HPP__
#define __VECTOR_OPS_HPP__

namespace cuv{

	enum ScalarFunctor{
		SF_EXP,
		SF_EXACT_EXP,
		SF_LOG,
		SF_SIGN,
		SF_SIGM,
		SF_EXACT_SIGM,
		SF_DSIGM,
		SF_TANH,
		SF_DTANH,
		SF_SQUARE,
		SF_SUBLIN,
		SF_ENERG,
		SF_INV,
		SF_SQRT,
		SF_NEGATE,

		SF_ADD,
		SF_SUBTRACT,
		SF_MULT,
		SF_DIV
	};


  template<class __vector_type>
  void
  apply_scalar_functor(__vector_type& v, const ScalarFunctor& sf);

  template<class __vector_type, class __value_type>
  void
  apply_scalar_functor(__vector_type& v, const ScalarFunctor& sf, const __value_type& param);


} // cuv

#endif
