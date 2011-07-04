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
	 *  @li SF_RSUB computes a - x
	 *  @li SF_MULT computes x * a
	 *  @li SF_DIV computes x / a
	 *  @li SF_RDIV computes a / x
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
		SF_RSUB,
		SF_MULT,
		SF_DIV,
		SF_RDIV,
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
	 * 	@li BF_AND computes  x && y
	 * 	@li BF_OR  computes  x || y
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
	 *  @li BF_SQSQLOSS computes  x*max(0,1-y-a)^2 + (1-x)*max(0,y-a)^2
	 *
	 *  With two scalar parameters a and b:
	 *  @li BF_AXPBY computes x = a * x + b * y
	 *
	 */
  enum BinaryFunctor{
	  // w/o params
	  BF_AND,
	  BF_OR,
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
	  BF_AXPBY,
	  BF_SQSQLOSS
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
  template<class __value_type, class __memory_space_type>
  void
  apply_0ary_functor(tensor<__value_type, __memory_space_type>& v, const NullaryFunctor& sf);

 /** 
  * @see apply_0ary_functor
  */
  template<class __value_type, class __memory_space_type>
  void apply_0ary_functor(tensor<__value_type, __memory_space_type, column_major>& v, const NullaryFunctor& sf){
      apply_0ary_functor(* reinterpret_cast<tensor<__value_type, __memory_space_type, row_major>* >(&v), sf);
  }
  /** 
   * @brief Apply a pointwise nullary functor with a scalar parameter to a vector.
   * 
   * @param v	Target vector 
   * @param sf	NullaryFunctor to apply 
   * @param param	scalar parameter 
   * 
   */
  template<class V1, class M>
  void
  apply_0ary_functor(tensor<V1, M>& v, const NullaryFunctor& sf, const V1& param);

 /** 
  * @see apply_0ary_functor
  */
  template<class V1, class M>
  void apply_0ary_functor(tensor<V1, M, column_major>& v, const NullaryFunctor& sf, const V1& param){
      apply_0ary_functor(* reinterpret_cast<tensor<V1, M, row_major>* >(&v), sf, param);
  }

  /** 
   * @brief Fill a vector with a sequence of numbers
   * 
   * @param v	Destination vector
   * 
   * This is a convenience wrapper that applies the nullary functor NF_SEQ to v.
   */
  template<class __value_type, class __memory_space_type, class __memory_layout_type>
  void sequence(tensor<__value_type, __memory_space_type, __memory_layout_type>& v){ apply_0ary_functor(v,NF_SEQ); }

  /** 
   * @brief Fill a vector with a value
   * 
   * @param v	Destination vector
   * @param p	Value to fill vector with
   * 
   * This is a convenience wrapper that applies the nullary functor NF_FILL to v.
   */
  template<class __value_type, class __memory_space_type, class __memory_layout_type, class S>
  void fill(tensor<__value_type, __memory_space_type, __memory_layout_type>& v, const S& p){
      apply_0ary_functor(v,NF_FILL,(__value_type)p);
  }


  /**
   * @defgroup scalar_functors Pointwise scalar functors
   *
   * @{
   */
  namespace detail{
	  /**
	   * These functions do the actual work for apply_scalar_functor and are instantiated in the .cu file.
	   *
	   * The operation performed is dst[i] = sf(src[i]) for all i
	   *
	   * @param dst   where we write to
	   * @param src   where we read from
	   * @param sf    the operation to be performed
	   * @param mask  whether the result should only applied to some values of dst
	   * @param numparams how many of the following parameters are specified
	   * @params p    first optional parameter
	   * @params p2   2nd optional parameter
	   */
	  template<class V1, class V2, class M, class S1, class S2>
	  void apply_scalar_functor(tensor<V1, M>&dst, const tensor<V2, M>&src, const ScalarFunctor& sf, const int& numparams=0, const tensor<unsigned char,M>* mask=NULL,const S1& p=S1(), const S2& p2=S2());

	  /**
	   * @see apply_scalar_functor
	   */
	  template<class V1, class V2, class M, class S1, class S2>
          void apply_scalar_functor(tensor<V1, M, column_major>& dst, const tensor<V2, M, column_major>& src, const ScalarFunctor& sf, const int& numparams=0, const tensor<unsigned char,M,column_major>* mask=NULL, const S1& p=S1(), const S2& p2=S2()){
              apply_scalar_functor(*reinterpret_cast<tensor<V1, M, row_major>* >(&dst), * reinterpret_cast<const tensor<V2, M, row_major>*>(&src), sf, numparams,reinterpret_cast<const tensor<unsigned char, M, row_major>*>(mask), p, p2); 
          }
  }

  /// @brief in-place, no parameters
  template<class D>
  void
  apply_scalar_functor(D& v, const ScalarFunctor& sf, const tensor<unsigned char,typename D::memory_space_type, typename D::memory_layout_type>* mask=NULL){
	  typedef typename D::value_type V;
	  detail::apply_scalar_functor(v,v,sf,0,mask,V(),V());
  }
  /// @brief no parameters
  template<class D, class S>
  void
  apply_scalar_functor(D& dst, const S& src, const ScalarFunctor& sf, const tensor<unsigned char,typename D::memory_space_type, typename D::memory_layout_type>*mask=NULL){
	  typedef typename S::value_type V;
	  detail::apply_scalar_functor(dst,src,sf,0,mask,V(),V());
  }

  /// @brief in-place, one parameter
  template<class D>
  void
  apply_scalar_functor(D& dst,const ScalarFunctor& sf, const typename D::value_type& p, const tensor<unsigned char,typename D::memory_space_type, typename D::memory_layout_type>*mask=NULL){
	  typedef typename D::value_type V;
	  detail::apply_scalar_functor(dst,dst,sf,1,mask,p,V());
  }
  /// @brief one parameter
  template<class D, class S>
  void
  apply_scalar_functor(D& dst,const S& src, const ScalarFunctor& sf, const typename S::value_type& p,const tensor<unsigned char,typename D::memory_space_type, typename D::memory_layout_type>*mask=NULL){
	  typedef typename S::value_type V;
	  detail::apply_scalar_functor(dst,src,sf,1,mask,p,V());
  }
  
  /// @brief in-place, two parameters
  template<class D>
  void
  apply_scalar_functor(D& dst, const ScalarFunctor& sf, const typename D::value_type& p, const typename D::value_type& p2, const tensor<unsigned char,typename D::memory_space_type, typename D::memory_layout_type>*mask=NULL){
	  detail::apply_scalar_functor(dst,dst,sf,2,mask,p,p2);
  }
  /// @brief two parameters
  template<class D, class S>
  void
  apply_scalar_functor(D& dst, const S& src, const ScalarFunctor& sf, const typename S::value_type& p, const typename S::value_type& p2, const tensor<unsigned char,typename D::memory_space_type, typename D::memory_layout_type>*mask=NULL){
	  detail::apply_scalar_functor(dst,src,sf,2,mask,p,p2);
  }

  /// @}

  /**
   * @defgroup binary_functors Pointwise binary functors
   *
   * @{
   */
  namespace detail{
	  /**
	   * These functions do the actual work for apply_binary_functor and are instantiated in the .cu file.
	   *
	   * The operation performed is dst[i] = bf(src1[i], src2[i]) for all i
	   *
	   * @param dst    where we write to
	   * @param src1   where we read from
	   * @param src2   where we read from
	   * @param bf    the operation to be performed
	   * @param numparams how many of the following parameters are specified
	   * @param p	  first optional parameter
	   * @param p2    2nd optional parameter
	   */
	  template<class V1, class V2, class V3, class M, class S1, class S2>
	  void apply_binary_functor(tensor<V1, M>& dst,const tensor<V2, M>& src1, const tensor<V3, M>&src2, const BinaryFunctor& bf, const int& numparams=0, const S1& p=S1(), const S2& p2=S2());
	  template<class V1, class V2, class V3, class M, class S1, class S2>
	  void apply_binary_functor(tensor<V1, M, column_major>& dst, const tensor<V2, M, column_major>& src1, const tensor<V3, M, column_major>& src2, const BinaryFunctor& bf, const int& numparams=0, const S1& p=S1(), const S2& p2=S2()){
              apply_binary_functor(*reinterpret_cast<tensor<V1, M, row_major>* >(&dst), * reinterpret_cast<const tensor<V2, M, row_major>*>(&src1), * reinterpret_cast<const tensor<V3, M, row_major>*>(&src2), bf, numparams, p, p2); 
          }
  }
  /// @brief in-place, no parameters
  template<class D, class S>
  void
  apply_binary_functor(D& v,  const S& w, const BinaryFunctor& bf){
	  typedef typename S::value_type V;
	  detail::apply_binary_functor(v,v,w,bf,0,V(),V());
  }
  /// @brief no parameters
  template<class D, class S, class S2>
  void
  apply_binary_functor(D& v,  const S& w, const S2& w2, const BinaryFunctor& bf){
	  typedef typename S::value_type V;
	  detail::apply_binary_functor(v,w,w2,bf,0,V(),V());
  }

  /// @brief in-place, one parameter
  template<class D, class S>
  void
  apply_binary_functor(D& v,const  S& w, const BinaryFunctor& bf, const typename S::value_type& param){
	  typedef typename S::value_type V;
	  detail::apply_binary_functor(v,v,w,bf,1,param,V());
  }
  /// @brief one parameter
  template<class D, class S, class S2>
  void
  apply_binary_functor(D& v,const  S& w, const S2& w2, const BinaryFunctor& bf, const typename S::value_type& param){
	  detail::apply_binary_functor(v,w,w2,bf,1,param,param);
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
  template<class __value_type, class __memory_space_type, class __memory_layout_type>
  void copy(tensor<__value_type, __memory_space_type, __memory_layout_type>& dst, const  tensor<__value_type, __memory_space_type, __memory_layout_type>& src){
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
  template<class __value_type, class __memory_space_type> bool has_inf(const tensor<__value_type, __memory_space_type>& v);
  /// @see has_inf
  template<class __value_type, class __memory_space_type> bool has_inf(const tensor<__value_type, __memory_space_type, column_major>& v){
	return has_inf(*reinterpret_cast<const tensor<__value_type,__memory_space_type>* >(&v));
  }
  /** 
   * @brief Check whether a float vector contains "NaN"
   * 
   * @param v Target vector 
   * 
   * @return true if v contains "NaN", false otherwise 
   */
  template<class __value_type, class __memory_space_type> bool has_nan(const tensor<__value_type, __memory_space_type>& v);
  /// @see has_nan
  template<class __value_type, class __memory_space_type> bool has_nan(const tensor<__value_type, __memory_space_type, column_major>& v){
	return has_nan(*reinterpret_cast<const tensor<__value_type,__memory_space_type>* >(&v));
  }
  /** 
   * @brief Count the elements with a certain scalar value in a vector
   * 
   * @param v vector
   * @param s scalar
   * 
   * @return count of elements with that scalar value
   */
  template<class __value_type, class __memory_space_type> unsigned int count(const tensor<__value_type, __memory_space_type>& v, const __value_type& s);
  /// @see sum
  template<class __value_type, class __memory_space_type> unsigned int count(const tensor<__value_type, __memory_space_type, column_major>& v, const __value_type& s){
	return count(*reinterpret_cast<const tensor<__value_type,__memory_space_type>* >(&v,s));
  }
  /** 
   * @brief Return the sum of a vector 
   * 
   * @param v vector
   * 
   * @return sum of v 
   */
  template<class __value_type, class __memory_space_type> float sum(const tensor<__value_type, __memory_space_type>& v);
  /// @see sum
  template<class __value_type, class __memory_space_type> float sum(const tensor<__value_type, __memory_space_type, column_major>& v){
	return sum(*reinterpret_cast<const tensor<__value_type,__memory_space_type>* >(&v));
  }
  /** 
   * @brief Return the two-norm or Euclidean norm of a vector 
   * 
   * @param v Target vector
   * 
   * @return Two-norm of v 
   */
  template<class __value_type, class __memory_space_type> float norm2(const tensor<__value_type, __memory_space_type>& v);
  /// @see norm2
  template<class __value_type, class __memory_space_type> float norm2(const tensor<__value_type, __memory_space_type, column_major>& v){
	return norm2(*reinterpret_cast<const tensor<__value_type,__memory_space_type>* >(&v));
  }
  /** 
   * @brief Return the one-norm or sum-norm of a vector 
   * 
   * @param v Target vector
   * 
   * @return one-norm of v 
   */
  template<class __value_type, class __memory_space_type> float norm1(const tensor<__value_type, __memory_space_type>& v);
  /// @see norm1
  template<class __value_type, class __memory_space_type> float norm1(const tensor<__value_type, __memory_space_type, column_major>& v){
	return norm1(*reinterpret_cast<const tensor<__value_type,__memory_space_type>* >(&v));
  }
  /** 
   * @brief Return the minimum entry of a vector 
   * 
   * @param v Target vector
   * 
   * @return Minimum entry of v 
   */
  template<class __value_type, class __memory_space_type> float minimum(const tensor<__value_type, __memory_space_type>& v);
  /// @see minimum
  template<class __value_type, class __memory_space_type> float minimum(const tensor<__value_type, __memory_space_type, column_major>& v){
	return minimum(*reinterpret_cast<const tensor<__value_type,__memory_space_type>* >(&v));
  }
  /** 
   * @brief Return the maximum entry of a vector 
   * 
   * @param v Target vector
   * 
   * @return Maximum entry of v 
   */
  template<class __value_type, class __memory_space_type> float maximum(const tensor<__value_type, __memory_space_type>& v);
  /// @see maximum
  template<class __value_type, class __memory_space_type> float maximum(const tensor<__value_type, __memory_space_type, column_major>& v){
	return maximum(*reinterpret_cast<const tensor<__value_type,__memory_space_type>* >(&v));
  }
  /** 
   * @brief Return the mean of the entries of a vector 
   * 
   * @param v Target vector
   * 
   * @return Mean of entries of v 
   */
  template<class __value_type, class __memory_space_type> float mean(const tensor<__value_type, __memory_space_type>& v);
  /// @see mean
  template<class __value_type, class __memory_space_type> float mean(const tensor<__value_type, __memory_space_type, column_major>& v){
	return mean(*reinterpret_cast<const tensor<__value_type,__memory_space_type>* >(&v));
  }
  /** 
   * @brief Return the variation of the entries of a vector 
   * 
   * @param v Target vector
   * 
   * @return Variation of entries of v 
   */
  template<class __value_type, class __memory_space_type> float var(const tensor<__value_type, __memory_space_type>& v);
  /// @see var
  template<class __value_type, class __memory_space_type> float var(const tensor<__value_type, __memory_space_type, column_major>& v){
	return var(*reinterpret_cast<const tensor<__value_type,__memory_space_type>* >(&v));
  }

  /** 
   * @brief Return the index of the maximum element
   * 
   * @param v Target vector
   * 
   * @return index of max element
   */
  template<class __value_type, class __memory_space_type> 
	  typename tensor<__value_type, __memory_space_type>::index_type 
	  arg_max(const tensor<__value_type, __memory_space_type>& v);
  /// @see arg_max
  template<class __value_type, class __memory_space_type> 
	  typename tensor<__value_type, __memory_space_type, column_major>::index_type 
	  arg_max(const tensor<__value_type, __memory_space_type, column_major>& v){
	return arg_max(*reinterpret_cast<const tensor<__value_type,__memory_space_type>* >(&v));
  }
  /** 
   * @brief Return the index of the minimum element
   * 
   * @param v Target vector
   * 
   * @return index of min element
   */
  template<class __value_type, class __memory_space_type> 
	  typename tensor<__value_type, __memory_space_type>::index_type 
	  arg_min(const tensor<__value_type, __memory_space_type>& v);
  /// @see arg_min
  template<class __value_type, class __memory_space_type> 
	  typename tensor<__value_type, __memory_space_type, column_major>::index_type 
	  arg_min(const tensor<__value_type, __memory_space_type, column_major>& v){
	return arg_min(*reinterpret_cast<const tensor<__value_type,__memory_space_type>* >(&v));
  }

 /** @} */ //end group reductions_vectors

} // cuv


 /* 
  * operator overloading for arithmatic operations on tensors
  */
  

/*
 * binary operators (tensor, scalar)
 */
  template<class T, class V, class M>
   cuv::tensor<T, V, M> 
    operator+ (const cuv::tensor<T, V, M>& v, const T& p){
        cuv::tensor<T, V, M> temp(v.shape());
        apply_scalar_functor(temp, v, cuv::SF_ADD, p);
        return temp;
  }
  template<class T, class V, class M>
   cuv::tensor<T, V, M> 
    operator- (const cuv::tensor<T, V, M>& v, const T& p){
        cuv::tensor<T, V, M> temp(v.shape());
        apply_scalar_functor(temp, v, cuv::SF_SUBTRACT, p);
        return temp;
  }
  template<class T, class V, class M>
   cuv::tensor<T, V, M> 
    operator* (const cuv::tensor<T, V, M>& v, const T& p){
        cuv::tensor<T, V, M> temp(v.shape());
        apply_scalar_functor(temp, v, cuv::SF_MULT, p);
        return temp;
  }
  template<class T, class V, class M>
   cuv::tensor<T, V, M> 
    operator/ (const cuv::tensor<T, V, M>& v, const T& p){
        cuv::tensor<T, V, M> temp(v.shape());
        apply_scalar_functor(temp, v, cuv::SF_DIV, p);
        return temp;
  }

/*
 * binary operators (scalar, tensor)
 */
  template<class T, class V, class M>
   cuv::tensor<T, V, M> 
    operator+ (const T& p, const cuv::tensor<T, V, M>& v){
        cuv::tensor<T, V, M> temp(v.shape());
        apply_scalar_functor(temp, v, cuv::SF_ADD, p);
        return temp;
  }
  template<class T, class V, class M>
   cuv::tensor<T, V, M> 
    operator- (const T& p, const cuv::tensor<T, V, M>& v){
        cuv::tensor<T, V, M> temp(v.shape());
        apply_scalar_functor(temp, v, cuv::SF_RSUB, p);
        return temp;
  }
  template<class T, class V, class M>
   cuv::tensor<T, V, M> 
    operator* (const T& p, const cuv::tensor<T, V, M>& v){
        cuv::tensor<T, V, M> temp(v.shape());
        apply_scalar_functor(temp, v, cuv::SF_MULT, p);
        return temp;
  }
  template<class T, class V, class M>
   cuv::tensor<T, V, M> 
    operator/ (const T& p, const cuv::tensor<T, V, M>& v){
        cuv::tensor<T, V, M> temp(v.shape());
        apply_scalar_functor(temp, v, cuv::SF_RDIV, p);
        return temp;
  }

/*
 * binary operators (tensor, tensor)
 */

  template<class T, class V, class M>
   cuv::tensor<T, V, M> 
    operator+ (const cuv::tensor<T, V, M>& v1, const cuv::tensor<T, V, M>& v2){
        cuv::tensor<T, V, M> temp(v1.shape());
        apply_binary_functor(temp, v1, v2, cuv::BF_ADD);
        return temp;
  }
  
  template<class T, class V, class M>
   cuv::tensor<T, V, M> 
    operator- (const cuv::tensor<T, V, M>& v1, const cuv::tensor<T, V, M>& v2){
        cuv::tensor<T, V, M> temp(v1.shape());
        apply_binary_functor(temp, v1, v2, cuv::BF_SUBTRACT);
        return temp;
  }
  template<class T, class V, class M>
   cuv::tensor<T, V, M> 
    operator* (const cuv::tensor<T, V, M>& v1, const cuv::tensor<T, V, M>& v2){
        cuv::tensor<T, V, M> temp(v1.shape());
        apply_binary_functor(temp, v1, v2, cuv::BF_MULT);
        return temp;
  }
  template<class T, class V, class M>
   cuv::tensor<T, V, M> 
    operator/ (const cuv::tensor<T, V, M>& v1, const cuv::tensor<T, V, M>& v2){
        cuv::tensor<T, V, M> temp(v1.shape());
        apply_binary_functor(temp, v1, v2, cuv::BF_DIV);
        return temp;
  }
        
/*
 * compound binary operators (tensor, tensor)
 */

  template<class T, class V, class M>
    cuv::tensor<T, V, M>& 
    operator-=(cuv::tensor<T, V, M>& v1, const cuv::tensor<T, V, M>& v2){
  	cuv::apply_binary_functor(v1,v2, cuv::BF_SUBTRACT);
  	return v1;
  }

  template<class T, class V, class M>
    cuv::tensor<T, V, M>& 
    operator*=(cuv::tensor<T, V, M>& v1, const cuv::tensor<T, V, M>& v2){
  	cuv::apply_binary_functor(v1,v2, cuv::BF_MULT);
  	return v1;
  }
  template<class T, class V, class M>
    cuv::tensor<T, V, M>& 
    operator/=(cuv::tensor<T, V, M>& v1, const cuv::tensor<T, V, M>& v2){
  	cuv::apply_binary_functor(v1,v2, cuv::BF_DIV);
  	return v1;
  }
  template<class T, class V, class M>
    cuv::tensor<T, V, M>& 
    operator+=(cuv::tensor<T, V, M>& v1, const cuv::tensor<T, V, M>& v2){
  	cuv::apply_binary_functor(v1,v2, cuv::BF_ADD);
  	return v1;
  }

/*
 * compound binary operators (tensor, scalar)
 */
 
  template<class T, class V, class M>
    cuv::tensor<T, V, M>& 
    operator-=(cuv::tensor<T, V, M>& v, const T& p){
  	cuv::apply_scalar_functor(v, cuv::SF_SUBTRACT, p);
  	return v;
  }
  template<class T, class V, class M>
    cuv::tensor<T, V, M>& 
    operator*=(cuv::tensor<T, V, M>& v, const T& p){
  	cuv::apply_scalar_functor(v, cuv::SF_MULT, p);
  	return v;
  }
  
  template<class T, class V, class M>
    cuv::tensor<T, V, M>& 
    operator/=(cuv::tensor<T, V, M>& v, const T& p){
  	cuv::apply_scalar_functor(v, cuv::SF_DIV, p);
  	return v;
  }
  template<class T, class V, class M>
    cuv::tensor<T, V, M>& 
    operator+=(cuv::tensor<T, V, M>& v, const T& p){
  	cuv::apply_scalar_functor(v, cuv::SF_ADD, p);
  	return v;
  }

/*
 * unary operators (tensor)
 */
  template<class T, class V, class M>
    cuv::tensor<T, V, M>
    operator-(const cuv::tensor<T, V, M>& v){
        cuv::tensor<T, V, M> temp(v.shape());
  	cuv::apply_scalar_functor(temp, v, cuv::SF_NEGATE);
  	return temp;
  }


#endif
