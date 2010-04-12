#ifndef __VECTOR_OPS_HPP__
#define __VECTOR_OPS_HPP__

namespace cuv{

	/** 
	 * @brief Scalar Functors for vectors and matrices
	 *  Applied pointwise to a vector/matrix.
	 *  Each entry x is transformed according to the given formular.
	 *
	 *  Without scalar parameters:
	 *
	 *	@li SF_EXP computes exp(x)
	 *	@li SF_LOG computes log(x)
	 * 	@li SF_SIGN computes sign(x)
	 * 	@li SF_SIGM computes 1/(1+exp(-x))
	 * 	@li SF_DSIGM computes x * (1-x)
	 * 	@li SF_TANH computes tanh(x)
	 *  @li SF_SQUARE computes x*x
	 *  @li SF_SUBLIN computes 1-x
	 *  @li SF_ENERG computes -log(x) 
	 *  @li SF_INV computes 1/x 
	 *  @li SF_SQRT computes sqrt(x)
	 *  @li SF_NEGATE computes -x
	 *  @li SF_ABS computes absolute value of x
	 *  @li SF_SMAX computes (1/x -1) * x
	 *
	 * 	With one scalar parameter a:
	 *  @li SF_ADD computes x + a
	 *  @li SF_SUBTRACT computes x - a
	 *  @li SF_MULT computes x * a
	 *	@li SF_DIV computes x / a
	 *	@li SF_MIN computes min(x,a)
	 *	@li SF_MAX computes max(x,a)
	 *
	 * 	With two scalar parameters a and b:
	 *
	 * 	@li SF_DTANH computes a/b * (a+x) + (a-x) 
	 */
	 
	enum ScalarFunctor{
		// w/o params
		SF_EXP,
		//SF_EXACT_EXP,
		SF_LOG,
		SF_SIGN,
		SF_SIGM,
		//SF_EXACT_SIGM,
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

	/** 
	 * @brief Binary functors for vectors and matrices
	 *  Applied pointwise to a vector/matrix.
	 *  The target entry x is calculated from the two source entries x,y according to the given formular.
	 *
	 *  Without scalar parameters:
	 *
	 * 	@li BF_ADD computes  x + y
	 * 	@li BF_SUBTRACT computes x - y
	 * 	@li BF_MULT computes x * y
	 * 	@li BF_DIV computes x / y
	 * 	@li BF_COPY computes y
	 * 	@li BF_MIN computes min(x,y)
	 * 	@li BF_MAX computes max(x,y)
	 *
	 *  With one scalar parameter a:
	 *  @li BF_AXPY computes a * x + y
	 *  @li BF_XPBY computes x + a * y
	 *
	 *  With two scalar parameters a and b:
	 *  @li BF_AXPBY computes a * x + b * y
	 *
	 */
  enum BinaryFunctor{
	  // w/o params
	  BF_ADD,
	  BF_SUBTRACT,
	  BF_MULT,
	  BF_DIV,
	  BF_COPY,
	  BF_MIN,
	  BF_MAX,

	  // w/ param
	  BF_AXPY,
	  BF_XPBY,
	  BF_AXPBY
  };

  /** 
   * @brief Nullary functors for vectors and matrices.
   * @li NF_FILL fills vector/matrix with parameter a
   * @li NF_SEQ fills vector/matrix with sequence of numbers starting from 1
   */
  enum NullaryFunctor{
	  NF_FILL,
	  NF_SEQ
  };

/** @defgroup functors_vectors Pointwise functors on vectors
 *   @{
 */

 /** 
  * @brief Apply a pointwise nullary functor to a vector.
  * 
  * @param v		Target vector 
  * @param sf 	NullaryFunctor to apply 
  * 
  */
  template<class __vector_type>
  void
  apply_0ary_functor(__vector_type& v, const NullaryFunctor& sf);

  /** 
   * @brief Apply a pointwise nullary functor with a scalar parameter to a vector.
   * 
   * @param v	Target vector 
   * @param sf	NullaryFunctor to apply 
   * @param param	scalar parameter 
   * 
   */
  template<class __vector_type, class __value_type>
  void
  apply_0ary_functor(__vector_type& v, const NullaryFunctor& sf, const __value_type& param);

  // convenience wrappers
  /** 
   * @brief Fill a vector with a sequence of numbers
   * 
   * @param v	Destination vector
   * 
   * This is a convenience wrapper that applies the nullary functor NF_SEQ to v.
   */
  template<class __vector_type>
  void sequence(__vector_type& v){ apply_0ary_functor(v,NF_SEQ); }

  /** 
   * @brief Fill a vector with a value
   * 
   * @param v	Destination vector
   * @param p	Value to fill vector with
   * 
   * This is a convenience wrapper that applies the nullary functor NF_FILL to v.
   */
  template<class __vector_type, class __value_type>
  void fill(__vector_type& v, const __value_type& p){ apply_0ary_functor(v,NF_FILL,p); }


  /** 
   * @brief Apply a pointwise unary functor to a vector
   * 
   * @param v Target vector 
   * @param sf ScalarFunctor to apply 
   * 
   */
  template<class __vector_type>
  void
  apply_scalar_functor(__vector_type& v, const ScalarFunctor& sf);



  /** 
   * @brief Apply pointwise unary functor with one scalar parameter to a vector
   * 
   * @param v Target vector
   * @param sf ScalarFunctor to apply
   * @param p scalar parameter
   * 
   */
  template<class __vector_type, class __value_type>
  void
  apply_scalar_functor(__vector_type& v, const ScalarFunctor& sf, const __value_type& p);
  /** 
   * @brief Apply pointwise unary functor with to scalar parameters to a vector
   * 
   * @param v Target vector
   * @param sf ScalarFunctor to apply 
   * @param p first scalar parameter 
   * @param p2 second scalar parameter
   */
  template<class __vector_type, class __value_type>
  void
  apply_scalar_functor(__vector_type& v, const ScalarFunctor& sf, const __value_type& p, const __value_type& p2);

  /** 
   * @brief Apply pointwise binary functor to a pair of matrices
   * 
   * @param v First parameter of binary functor,  destination vector
   * @param w Second parameter of binary functor 
   * @param bf BinaryFunctor to apply
   * 
   */
  template<class __vector_type1, class __vector_type2>
  void
  apply_binary_functor(__vector_type1& v, __vector_type2& w, const BinaryFunctor& bf);

  /** 
   * @brief Apply pointwise binary functor with one scalar parameter to a pair of matrices 
   * 
   * @param v	First parameter of binary functor, destination vector 
   * @param w	Second parameter of binary functor 
   * @param bf	 BinaryFunctor to apply
   * @param param Scalar parameter 
   */
  template<class __vector_type1, class __vector_type2, class __value_type>
  void
  apply_binary_functor(__vector_type1& v, __vector_type2& w, const BinaryFunctor& bf, const __value_type& param);

  /** 
   * @brief Apply pointwise binary functor with two scalar parameters to a pair of matrices 
   * 
   * @param v	First parameter of binary functor, destination vector 
   * @param w	Second parameter of binary functor 
   * @param bf	 BinaryFunctor to apply
   * @param param First scalar parameter 
   * @param param2 Secont scalar parameter 
   *
   */
  template<class __vector_type1, class __vector_type2, class __value_type>
  void
  apply_binary_functor(__vector_type1& v, __vector_type2& w, const BinaryFunctor& bf, const __value_type& param, const __value_type& param2);

  /** 
   * @brief Copy one vector into another. 
   * 
   * @param dst Destination vector
   * @param src	Source vector 
   * 
   * This is a convenience wrapper that applies the binary functor BF_COPY 
   */
  template<class __vector_type>
  void copy(__vector_type& dst, __vector_type& src){
	  apply_binary_functor(dst,src,BF_COPY);
  }
 /** @} */ //end group functors_vectors

/** @defgroup reductions_vectors Functors reducing a vector to a scalar
 *   @{
 */

  /** 
   * @brief Check whether a float vector contains "Inf" or "-Inf"
   * 
   * @param v Target vector 
   * 
   * @return true if v contains "Inf" or "-Inf", false otherwise 
   */
  template<class __vector_type1> bool has_inf(__vector_type1& v);
  /** 
   * @brief Check whether a float vector contains "NaN"
   * 
   * @param v Target vector 
   * 
   * @return true if v contains "NaN", false otherwise 
   */
  template<class __vector_type1> bool has_nan(__vector_type1& v);
  /** 
   * @brief Return the two-norm or Euclidean norm of a vector 
   * 
   * @param v Target vector
   * 
   * @return Two-norm of v 
   */
  template<class __vector_type1> float norm2(__vector_type1& v);
  /** 
   * @brief Return the one-norm or sum-norm of a vector 
   * 
   * @param v Target vector
   * 
   * @return one-norm of v 
   */
  template<class __vector_type1> float norm1(__vector_type1& v);
  /** 
   * @brief Return the minimum entry of a vector 
   * 
   * @param v Target vector
   * 
   * @return Minimum entry of v 
   */
  template<class __vector_type1> float minimum(__vector_type1& v);
  /** 
   * @brief Return the maximum entry of a vector 
   * 
   * @param v Target vector
   * 
   * @return Maximum entry of v 
   */
  template<class __vector_type1> float maximum(__vector_type1& v);
  /** 
   * @brief Return the mean of the entries of a vector 
   * 
   * @param v Target vector
   * 
   * @return Mean of entries of v 
   */
  template<class __vector_type1> float mean(__vector_type1& v);
  /** 
   * @brief Return the variation of the entries of a vector 
   * 
   * @param v Target vector
   * 
   * @return Variation of entries of v 
   */
  template<class __vector_type1> float var(__vector_type1& v);

 /** @} */ //end group reductions_vectors

} // cuv

#endif
