#ifndef __VECTOR_OPS_HPP__
#define __VECTOR_OPS_HPP__

namespace cuv{

	enum ScalarFunctor{
		// w/o params
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
		SF_ABS,
		SF_SMAX,

		// with param
		SF_ADD,
		SF_SUBTRACT,
		SF_MULT,
		SF_DIV,
		SF_MIN,
		SF_MAX
	};

  enum BinaryFunctor{
	  // w/o params
	  BF_ADD,
	  BF_SUBTRACT,
	  BF_MULT,
	  BF_DIV,
	  BF_COPY,

	  // w/ param
	  BF_AXPY,
	  BF_XPBY,
	  BF_AXPBY
  };

  enum NullaryFunctor{
	  NF_FILL,
	  NF_SEQ
  };

  /*
   * Pointwise Null-ary Functor
   * v = sf()
   */
  template<class __vector_type>
  void
  apply_0ary_functor(__vector_type& v, const NullaryFunctor& sf);

  template<class __vector_type, class __value_type>
  void
  apply_0ary_functor(__vector_type& v, const NullaryFunctor& sf, const __value_type& param);

  // convenience wrappers
  template<class __vector_type>
  void sequence(__vector_type& v){ apply_0ary_functor(v,NF_SEQ); }
  template<class __vector_type, class __value_type>
  void fill(__vector_type& v, const __value_type& p){ apply_0ary_functor(v,NF_FILL,p); }


  /*
   * Pointwise Unary Functor
   * v = sf(v)
   */
  template<class __vector_type>
  void
  apply_scalar_functor(__vector_type& v, const ScalarFunctor& sf);



  /*
   * Pointwise Unary Functor with scalar parameter
   * v = sf(v, param)
   */
  template<class __vector_type, class __value_type>
  void
  apply_scalar_functor(__vector_type& v, const ScalarFunctor& sf, const __value_type& param);
  /*
   * Pointwise Unary Functor with scalar parameter
   * v = sf(v, param, param2)
   */
  template<class __vector_type, class __value_type>
  void
  apply_scalar_functor(__vector_type& v, const ScalarFunctor& sf, const __value_type& param, const __value_type& param2);

  /*
   * Pointwise Binary Functor
   * v = bf(v,w)
   */
  template<class __vector_type1, class __vector_type2>
  void
  apply_binary_functor(__vector_type1& v, __vector_type2& w, const BinaryFunctor& bf);

  /*
   * Pointwise Binary Functor
   * v = bf(v,w,param)
   */
  template<class __vector_type1, class __vector_type2, class __value_type>
  void
  apply_binary_functor(__vector_type1& v, __vector_type2& w, const BinaryFunctor& bf, const __value_type& param);

  /*
   * Pointwise Binary Functor
   * v = bf(v,w,param,param2)
   */
  template<class __vector_type1, class __vector_type2, class __value_type>
  void
  apply_binary_functor(__vector_type1& v, __vector_type2& w, const BinaryFunctor& bf, const __value_type& param, const __value_type& param2);

  // convenience wrappers
  template<class __vector_type>
  void copy(__vector_type& dst, __vector_type& src){
	  apply_binary_functor(dst,src,BF_COPY);
  }


  /*
   * reductions
   *
   */
  template<class __vector_type1> bool has_inf(__vector_type1& v);
  template<class __vector_type1> bool has_nan(__vector_type1& v);
  template<class __vector_type1> float norm2(__vector_type1& v);
  template<class __vector_type1> float norm1(__vector_type1& v);
  template<class __vector_type1> float mean(__vector_type1& v);
  template<class __vector_type1> float var(__vector_type1& v);


} // cuv

#endif
