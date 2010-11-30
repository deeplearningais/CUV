//*LB*
// Copyright (c) 2010, University of Bonn, Institute for Computer Science VI
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of the University of Bonn 
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*LE*





#ifndef __VECTOR_OPS_HPP__
#define __VECTOR_OPS_HPP__

namespace cuv{
        template<class T, class V, class I>
        class vector;


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
		SF_POSLIN,
		// rectifying transfer function
		SF_RECT,
		SF_DRECT,

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
	 * 	@li BF_ADD computes  x += y
	 * 	@li BF_SUBTRACT computes x -= y
	 * 	@li BF_MULT computes x *= y
	 * 	@li BF_DIV computes x /= y
	 * 	@li BF_COPY computes x = y
	 * 	@li BF_MIN computes x = min(x,y)
	 * 	@li BF_MAX computes x = max(x,y)
	 *
	 *  With one scalar parameter a:
	 *  @li BF_AXPY computes x = a * x + y
	 *  @li BF_XPBY computes x += a * y
	 *
	 *  With two scalar parameters a and b:
	 *  @li BF_AXPBY computes x = a * x + b * y
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
  apply_binary_functor(__vector_type1& v,  const __vector_type2& w, const BinaryFunctor& bf);

  /** 
   * @brief Apply pointwise binary functor with one scalar parameter to a pair of matrices 
   * 
   * @param v	First parameter of binary functor, destination vector 
   * @param w	Second parameter of binary functor 
   * @param bf	 BinaryFunctor to apply
   * @param param Scalar parameter and .hpp
   */
  template<class __vector_type1, class __vector_type2, class __value_type>
  void
  apply_binary_functor(__vector_type1& v,const  __vector_type2& w, const BinaryFunctor& bf, const __value_type& param);

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
  apply_binary_functor(__vector_type1& v, const __vector_type2& w, const BinaryFunctor& bf, const __value_type& param, const __value_type& param2);

  /** 
   * @brief Copy one vector into another. 
   * 
   * @param dst Destination vector
   * @param src	Source vector 
   * 
   * This is a convenience wrapper that applies the binary functor BF_COPY 
   */
  template<class __vector_type>
  void copy(__vector_type& dst, const  __vector_type& src){
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
  template<class __vector_type1> bool has_inf(const __vector_type1& v);
  /** 
   * @brief Check whether a float vector contains "NaN"
   * 
   * @param v Target vector 
   * 
   * @return true if v contains "NaN", false otherwise 
   */
  template<class __vector_type1> bool has_nan(const __vector_type1& v);
  /** 
   * @brief Return the sum of a vector 
   * 
   * @param v vector
   * 
   * @return sum of v 
   */
  template<class __vector_type1> float sum(const __vector_type1& v);
  /** 
   * @brief Return the two-norm or Euclidean norm of a vector 
   * 
   * @param v Target vector
   * 
   * @return Two-norm of v 
   */
  template<class __vector_type1> float norm2(const __vector_type1& v);
  /** 
   * @brief Return the one-norm or sum-norm of a vector 
   * 
   * @param v Target vector
   * 
   * @return one-norm of v 
   */
  template<class __vector_type1> float norm1(const __vector_type1& v);
  /** 
   * @brief Return the minimum entry of a vector 
   * 
   * @param v Target vector
   * 
   * @return Minimum entry of v 
   */
  template<class __vector_type1> float minimum(const __vector_type1& v);
  /** 
   * @brief Return the maximum entry of a vector 
   * 
   * @param v Target vector
   * 
   * @return Maximum entry of v 
   */
  template<class __vector_type1> float maximum(const __vector_type1& v);
  /** 
   * @brief Return the mean of the entries of a vector 
   * 
   * @param v Target vector
   * 
   * @return Mean of entries of v 
   */
  template<class __vector_type1> float mean(const __vector_type1& v);
  /** 
   * @brief Return the variation of the entries of a vector 
   * 
   * @param v Target vector
   * 
   * @return Variation of entries of v 
   */
  template<class __vector_type1> float var(const __vector_type1& v);

  /** 
   * @brief Return the index of the maximum element
   * 
   * @param v Target vector
   * 
   * @return index of max element
   */
  template<class __vector_type1> 
	  typename __vector_type1::index_type 
	  arg_max(const __vector_type1& v);
  /** 
   * @brief Return the index of the minimum element
   * 
   * @param v Target vector
   * 
   * @return index of min element
   */
  template<class __vector_type1> 
	  typename __vector_type1::index_type 
	  arg_min(const __vector_type1& v);

 /** @} */ //end group reductions_vectors

} // cuv


 /* 
  * operator overloading for arithmatic operations on vectors
  */
  
  
/*  template<class T, class V, class I>
   cuv::vector<T, V, I> 
    operator- (const cuv::vector<T, V, I>& v1, const cuv::vector<T, V, I>& v2){
        cuv::vector<T, V, I> temp= v1;
        temp-= v2;
        return temp;
  }*/
  
  template<class T, class V, class I>
   cuv::vector<T, V, I> 
    operator+ (const cuv::vector<T, V, I>& v, const V p){
        cuv::vector<T, V, I> temp = v;
        temp+= p;
        return temp;
  }
  template<class T, class V, class I>
   cuv::vector<T, V, I> 
    operator- (const cuv::vector<T, V, I>& v, const V p){
        cuv::vector<T, V, I> temp = v;
        temp-= p;
        return temp;
  }
  template<class T, class V, class I>
   cuv::vector<T, V, I> 
    operator* (const cuv::vector<T, V, I>& v, const V p){
        cuv::vector<T, V, I> temp = v;
        temp*= p;
        return temp;
  }
  template<class T, class V, class I>
   cuv::vector<T, V, I> 
    operator/ (const cuv::vector<T, V, I>& v, const V p){
        cuv::vector<T, V, I> temp = v;
        temp/= p;
        return temp;
  }
  template<class T, class V, class I>
   cuv::vector<T, V, I> 
    operator+ (const cuv::vector<T, V, I>& v1, const cuv::vector<T, V, I>& v2){
        cuv::vector<T, V, I> temp = v1;
        temp+= v2;
        return temp;
  }
  
  template<class T, class V, class I>
   cuv::vector<T, V, I> 
    operator- (const cuv::vector<T, V, I>& v1, const cuv::vector<T, V, I>& v2){
        cuv::vector<T, V, I> temp = v1;
        temp-= v2;
        return temp;
  }
  template<class T, class V, class I>
   cuv::vector<T, V, I> 
    operator* (const cuv::vector<T, V, I>& v1, const cuv::vector<T, V, I>& v2){
        cuv::vector<T, V, I> temp = v1;
        temp*= v2;
        return temp;
  }
  template<class T, class V, class I>
   cuv::vector<T, V, I> 
    operator/ (const cuv::vector<T, V, I>& v1, const cuv::vector<T, V, I>& v2){
        cuv::vector<T, V, I> temp = v1;
        temp/= v2;
        return temp;
  }
        
  template<class T, class V, class I>
    cuv::vector<T, V, I>& 
    operator-=(cuv::vector<T, V, I>& v1, const cuv::vector<T, V, I>& v2){
  	cuv::apply_binary_functor(v1,v2, cuv::BF_SUBTRACT);
  	return v1;
  }

  template<class T, class V, class I>
    cuv::vector<T, V, I>& 
    operator*=(cuv::vector<T, V, I>& v1, const cuv::vector<T, V, I>& v2){
  	cuv::apply_binary_functor(v1,v2, cuv::BF_MULT);
  	return v1;
  }
  template<class T, class V, class I>
    cuv::vector<T, V, I>& 
    operator/=(cuv::vector<T, V, I>& v1, const cuv::vector<T, V, I>& v2){
  	cuv::apply_binary_functor(v1,v2, cuv::BF_DIV);
  	return v1;
  }
  template<class T, class V, class I>
    cuv::vector<T, V, I>& 
    operator+=(cuv::vector<T, V, I>& v1, const cuv::vector<T, V, I>& v2){
  	cuv::apply_binary_functor(v1,v2, cuv::BF_ADD);
  	return v1;
  }
 
  template<class T, class V, class I>
    cuv::vector<T, V, I>& 
    operator-=(cuv::vector<T, V, I>& v, const T& p){
  	cuv::apply_scalar_functor(v, cuv::SF_SUBTRACT, p);
  	return v;
  }
  template<class T, class V, class I>
    cuv::vector<T, V, I>& 
    operator*=(cuv::vector<T, V, I>& v, const T& p){
  	cuv::apply_scalar_functor(v, cuv::SF_MULT, p);
  	return v;
  }
  
  template<class T, class V, class I>
    cuv::vector<T, V, I>& 
    operator/=(cuv::vector<T, V, I>& v, const T& p){
  	cuv::apply_scalar_functor(v, cuv::SF_DIV, p);
  	return v;
  }
  template<class T, class V, class I>
    cuv::vector<T, V, I>& 
    operator+=(cuv::vector<T, V, I>& v, const T& p){
  	cuv::apply_scalar_functor(v, cuv::SF_ADD, p);
  	return v;
  }


#endif
