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





#ifndef __TENSOR_OPS_HPP__
#define __TENSOR_OPS_HPP__

#include <cuv/basics/tensor.hpp>

namespace cuv{


	/** 
	 * @brief Scalar Functors for vectors and matrices
	 *  Applied pointwise to a vector/matrix.
	 *  Each entry x is transformed according to the given formular.
	 *
	 *  Without scalar parameters:
	 *
	 *  @li SF_COPY computes x (the identity)
	 *  @li SF_EXP computes exp(x)
	 *  @li SF_LOG computes log(x)
	 *  @li SF_SIGN computes sign(x)
	 *  @li SF_SIGM computes 1/(1+exp(-x))
	 *  @li SF_DSIGM computes x * (1-x)
	 *  @li SF_TANH computes tanh(x)
	 *  @li SF_SQUARE computes x*x
	 *  @li SF_SUBLIN computes 1-x
	 *  @li SF_ENERG computes -log(x) 
	 *  @li SF_INV computes 1/x 
	 *  @li SF_SQRT computes sqrt(x)
	 *  @li SF_NEGATE computes -x
	 *  @li SF_ABS computes absolute value of x
	 *  @li SF_SMAX computes (1/x -1) * x
	 *
	 * With one scalar parameter a:
	 *  @li SF_ADD computes x + a
	 *  @li SF_SUBTRACT computes x - a
	 *  @li SF_MULT computes x * a
	 *  @li SF_DIV computes x / a
	 *  @li SF_MIN computes min(x,a)
	 *  @li SF_MAX computes max(x,a)
	 *  @li SF_EQ computes x == a
	 *  @li SF_LT computes x < a
	 *  @li SF_GT computes x > a
	 *  @li SF_LEQ computes x <= a
	 *  @li SF_GEQ computes x >= a
	 *
	 * With two scalar parameters a and b:
	 *
	 *  @li SF_DTANH computes a/b * (a+x) + (a-x) 
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
		SF_COPY,

		// with param
		SF_ADD,
		SF_SUBTRACT,
		SF_MULT,
		SF_DIV,
		SF_MIN,
		SF_MAX,
		SF_EQ,
		SF_LT,
		SF_GT,
		SF_LEQ,
		SF_GEQ
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
	  BF_MIN,
	  BF_MAX,
	  BF_ATAN2,
	  BF_NORM,

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
  template<class __vector_type>
  void
  apply_0ary_functor(__vector_type& v, const NullaryFunctor& sf, const typename __vector_type::value_type& param);

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
  template<class __vector_type>
  void fill(__vector_type& v, const typename __vector_type::value_type& p){ apply_0ary_functor(v,NF_FILL,p); }


  /**
   * @defgroup scalar_functors Pointwise scalar functors
   *
   * @{
   */
  namespace detail{
	  template<class D, class S, class V>
	  void
	  apply_scalar_functor(D&,const S&, const ScalarFunctor& sf, const int& numparams=0, const V& p=V(), const V& p2=V());
  }

  /// @brief in-place, no parameters
  template<class D>
  void
  apply_scalar_functor(D& v, const ScalarFunctor& sf){
	  typedef typename D::value_type V;
	  detail::apply_scalar_functor<D,D,V>(v,v,sf);
  }
  /// @brief no parameters
  template<class D, class S>
  void
  apply_scalar_functor(D& dst, const S& src, const ScalarFunctor& sf){
	  typedef typename S::value_type V;
	  detail::apply_scalar_functor<D,S,V>(dst,src,sf);
  }

  /// @brief in-place, one parameter
  template<class D>
  void
  apply_scalar_functor(D& dst,const ScalarFunctor& sf, const typename D::value_type& p){
	  detail::apply_scalar_functor(dst,dst,sf,1,p);
  }
  /// @brief one parameter
  template<class D, class S>
  void
  apply_scalar_functor(D& dst,const S& src, const ScalarFunctor& sf, const typename S::value_type& p){
	  detail::apply_scalar_functor(dst,src,sf,1,p);
  }
  
  /// @brief in-place, two parameters
  template<class D>
  void
  apply_scalar_functor(D& dst, const ScalarFunctor& sf, const typename D::value_type& p, const typename D::value_type& p2){
	  detail::apply_scalar_functor(dst,dst,sf,2,p,p2);
  }
  /// @brief two parameters
  template<class D, class S>
  void
  apply_scalar_functor(D& dst, const S& src, const ScalarFunctor& sf, const typename S::value_type& p, const typename S::value_type& p2){
	  detail::apply_scalar_functor(dst,src,sf,2,p,p2);
  }

  /// @}

  /**
   * @defgroup binary_functors Pointwise binary functors
   *
   * @{
   */
  namespace detail{
	  template<class D, class S, class S2, class V>
	  void
	  apply_binary_functor(D&,const S&, const S2&, const BinaryFunctor& bf, const int& numparams=0, const V& p=V(), const V& p2=V());
  }
  /// @brief in-place, no parameters
  template<class D, class S>
  void
  apply_binary_functor(D& v,  const S& w, const BinaryFunctor& bf){
	  typedef typename S::value_type V;
	  detail::apply_binary_functor<D,D,S,V>(v,v,w,bf);
  }
  /// @brief no parameters
  template<class D, class S, class S2>
  void
  apply_binary_functor(D& v,  const S& w, const S2& w2, const BinaryFunctor& bf){
	  typedef typename S::value_type V;
	  detail::apply_binary_functor<D,S,S2,V>(v,w,w2,bf);
  }

  /// @brief in-place, one parameter
  template<class D, class S>
  void
  apply_binary_functor(D& v,const  S& w, const BinaryFunctor& bf, const typename S::value_type& param){
	  detail::apply_binary_functor(v,v,w,bf,1,param);
  }
  /// @brief one parameter
  template<class D, class S, class S2>
  void
  apply_binary_functor(D& v,const  S& w, const S2& w2, const BinaryFunctor& bf, const typename S::value_type& param){
	  detail::apply_binary_functor(v,w,w2,bf,1,param);
  }

  /// @brief in-place, two parameters
  template<class D, class S>
  void
  apply_binary_functor(D& v, const S& w, const BinaryFunctor& bf, const typename S::value_type& param, const typename S::value_type& param2){
	  detail::apply_binary_functor(v,v,w,bf,2,param,param2);
  }
  /// @brief two parameters
  template<class D, class S, class S2>
  void
  apply_binary_functor(D& v, const S& w, const S2& w2, const BinaryFunctor& bf, const typename S::value_type& param, const typename S::value_type& param2){
	  detail::apply_binary_functor(v,w,w2,bf,2,param,param2);
  }
  /// @}

  /** 
   * @brief Copy one vector into another. 
   * 
   * @param dst Destination vector
   * @param src	Source vector 
   * 
   * This is a convenience wrapper that applies the binary functor SF_COPY 
   */
  template<class __vector_type>
  void copy(__vector_type& dst, const  __vector_type& src){
	  apply_scalar_functor(dst,src,SF_COPY);
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
  
  
/*  template<class T, class V>
   cuv::tensor<T, V> 
    operator- (const cuv::tensor<T, V>& v1, const cuv::tensor<T, V>& v2){
        cuv::tensor<T, V> temp= v1;
        temp-= v2;
        return temp;
  }*/
  
  template<class T, class V>
   cuv::tensor<T, V> 
    operator+ (const cuv::tensor<T, V>& v, const V p){
        cuv::tensor<T, V> temp = v;
        temp+= p;
        return temp;
  }
  template<class T, class V>
   cuv::tensor<T, V> 
    operator- (const cuv::tensor<T, V>& v, const V p){
        cuv::tensor<T, V> temp = v;
        temp-= p;
        return temp;
  }
  template<class T, class V>
   cuv::tensor<T, V> 
    operator* (const cuv::tensor<T, V>& v, const V p){
        cuv::tensor<T, V> temp = v;
        temp*= p;
        return temp;
  }
  template<class T, class V>
   cuv::tensor<T, V> 
    operator/ (const cuv::tensor<T, V>& v, const V p){
        cuv::tensor<T, V> temp = v;
        temp/= p;
        return temp;
  }
  template<class T, class V>
   cuv::tensor<T, V> 
    operator+ (const cuv::tensor<T, V>& v1, const cuv::tensor<T, V>& v2){
        cuv::tensor<T, V> temp = v1;
        temp+= v2;
        return temp;
  }
  
  template<class T, class V>
   cuv::tensor<T, V> 
    operator- (const cuv::tensor<T, V>& v1, const cuv::tensor<T, V>& v2){
        cuv::tensor<T, V> temp = v1;
        temp-= v2;
        return temp;
  }
  template<class T, class V>
   cuv::tensor<T, V> 
    operator* (const cuv::tensor<T, V>& v1, const cuv::tensor<T, V>& v2){
        cuv::tensor<T, V> temp = v1;
        temp*= v2;
        return temp;
  }
  template<class T, class V>
   cuv::tensor<T, V> 
    operator/ (const cuv::tensor<T, V>& v1, const cuv::tensor<T, V>& v2){
        cuv::tensor<T, V> temp = v1;
        temp/= v2;
        return temp;
  }
        
  template<class T, class V>
    cuv::tensor<T, V>& 
    operator-=(cuv::tensor<T, V>& v1, const cuv::tensor<T, V>& v2){
  	cuv::apply_binary_functor(v1,v2, cuv::BF_SUBTRACT);
  	return v1;
  }

  template<class T, class V>
    cuv::tensor<T, V>& 
    operator*=(cuv::tensor<T, V>& v1, const cuv::tensor<T, V>& v2){
  	cuv::apply_binary_functor(v1,v2, cuv::BF_MULT);
  	return v1;
  }
  template<class T, class V>
    cuv::tensor<T, V>& 
    operator/=(cuv::tensor<T, V>& v1, const cuv::tensor<T, V>& v2){
  	cuv::apply_binary_functor(v1,v2, cuv::BF_DIV);
  	return v1;
  }
  template<class T, class V>
    cuv::tensor<T, V>& 
    operator+=(cuv::tensor<T, V>& v1, const cuv::tensor<T, V>& v2){
  	cuv::apply_binary_functor(v1,v2, cuv::BF_ADD);
  	return v1;
  }
 
  template<class T, class V>
    cuv::tensor<T, V>& 
    operator-=(cuv::tensor<T, V>& v, const T& p){
  	cuv::apply_scalar_functor(v, cuv::SF_SUBTRACT, p);
  	return v;
  }
  template<class T, class V>
    cuv::tensor<T, V>& 
    operator*=(cuv::tensor<T, V>& v, const T& p){
  	cuv::apply_scalar_functor(v, cuv::SF_MULT, p);
  	return v;
  }
  
  template<class T, class V>
    cuv::tensor<T, V>& 
    operator/=(cuv::tensor<T, V>& v, const T& p){
  	cuv::apply_scalar_functor(v, cuv::SF_DIV, p);
  	return v;
  }
  template<class T, class V>
    cuv::tensor<T, V>& 
    operator+=(cuv::tensor<T, V>& v, const T& p){
  	cuv::apply_scalar_functor(v, cuv::SF_ADD, p);
  	return v;
  }


#endif
