#ifndef __MATRIX_OPS_HPP__
#define __MATRIX_OPS_HPP__

#include <vector_ops/vector_ops.hpp>
#include <basics/dev_dense_matrix.hpp>
#include <basics/host_dense_matrix.hpp>

namespace cuv{

  /****************************************
   * Wrappers for Vector Ops
   ****************************************
   */

 /** @defgroup functors_matrices Pointwise Functors on Matrices
  * @{
  */

  /** 
   * @brief Apply a pointwise nullary functor to a matrix.
   * 
   * @param m		Target matrix 
   * @param sf 	NullaryFunctor to apply 
   * 
   */
  template<class V,class M,class I> 
	  void apply_0ary_functor(dev_dense_matrix<V,M,I>& m, const NullaryFunctor& sf){ apply_0ary_functor(m.vec(),sf); }
  /** 
   * @brief Apply a pointwise nullary functor with a scalar parameter to a matrix.
   * 
   * @param m	Target matrix 
   * @param sf	NullaryFunctor to apply 
   * @param param	scalar parameter 
   * 
   */
  template<class V,class M,class I,class P> 
	  void apply_0ary_functor(dev_dense_matrix<V,M,I>& m, const NullaryFunctor& sf, const P& param){apply_0ary_functor(m.vec(),sf,param);}
  /**
   * @copydoc apply_0ary_functor(dev_dense_matrix<V,M,I>& m, const NullaryFunctor& sf)
   */
  template<class V,class M,class I> 
	  void apply_0ary_functor(host_dense_matrix<V,M,I>& m, const NullaryFunctor& sf){ apply_0ary_functor(m.vec(),sf); }
  /**
   * @copydoc apply_0ary_functor(dev_dense_matrix<V,M,I>& m, const NullaryFunctor& sf,const P& param)
   */
  template<class V,class M,class I,class P> 
	  void apply_0ary_functor(host_dense_matrix<V,M,I>& m, const NullaryFunctor& sf, const P& param){apply_0ary_functor(m.vec(),sf,param);}

  /** 
   * @brief Apply a pointwise unary functor to a matrix
   * 
   * @param m Target matrix 
   * @param sf ScalarFunctor to apply 
   * 
   */
  template<class V,class M,class I> 
	  void apply_scalar_functor(dev_dense_matrix<V,M,I>& m, const ScalarFunctor& sf){apply_scalar_functor(m.vec(),sf);}
  /**
   * @copydoc apply_scalar_functor(dev_dense_matrix<V,M,I>& m, const ScalarFunctor& sf)
   */
  template<class V,class M,class I> 
	  void apply_scalar_functor(host_dense_matrix<V,M,I>& m, const ScalarFunctor& sf){apply_scalar_functor(m.vec(),sf);}

  /** 
   * @brief Apply pointwise unary functor with one scalar parameter to a matrix
   * 
   * @param m Target matrix
   * @param sf ScalarFunctor to apply
   * @param p scalar parameter
   * 
   */
  template<class V,class M,class I,class P> 
	  void apply_scalar_functor(dev_dense_matrix<V,M,I>& m, const ScalarFunctor& sf, const P& p) {apply_scalar_functor(m.vec(),sf,p);};
  /**
   * @copydoc apply_scalar_functor(dev_dense_matrix<V,M,I>& m, const ScalarFunctor& sf, const P& p)
   */
  template<class V,class M,class I,class P> 
	  void apply_scalar_functor(host_dense_matrix<V,M,I>& m, const ScalarFunctor& sf, const P& p) {apply_scalar_functor(m.vec(),sf,p);};

  /** 
   * @brief Apply pointwise unary functor with to scalar parameters to a matrix
   * 
   * @param m Target matrix
   * @param sf ScalarFunctor to apply 
   * @param p first scalar parameter 
   * @param p2 second scalar parameter
   */
  template<class V,class M,class I,class P>
	  void apply_scalar_functor(dev_dense_matrix<V,M,I>& m, const ScalarFunctor& sf, const P& p, const P& p2) {apply_scalar_functor(m.vec(),sf,p,p2);};
  /**
   * @copydoc apply_scalar_functor(dev_dense_matrix<V,M,I>& m, const ScalarFunctor& sf, const P& p, const P& p2) 
   */
  template<class V,class M,class I,class P>
	  void apply_scalar_functor(host_dense_matrix<V,M,I>& m, const ScalarFunctor& sf, const P& p, const P& p2) {apply_scalar_functor(m.vec(),sf,p,p2);};

  /** 
   * @brief Apply pointwise binary functor to a pair of matrices
   * 
   * @param v First parameter of binary functor,  destination matrix
   * @param w Second parameter of binary functor 
   * @param bf BinaryFunctor to apply
   * 
   */
  template<class V,class M,class I,class V2> 
	  void apply_binary_functor(dev_dense_matrix<V,M,I>& v, dev_dense_matrix<V2,M,I>& w, const BinaryFunctor& bf){apply_binary_functor(v.vec(),w.vec(),bf);}
  /**
   * @copydoc apply_binary_functor(dev_dense_matrix<V,M,I>& v, dev_dense_matrix<V2,M,I>& w, const BinaryFunctor& bf)
   */
  template<class V,class M,class I,class V2> 
	  void apply_binary_functor(host_dense_matrix<V,M,I>& v, host_dense_matrix<V2,M,I>& w, const BinaryFunctor& bf){apply_binary_functor(v.vec(),w.vec(),bf);}

  /** 
   * @brief Apply pointwise binary functor with one scalar parameter to a pair of matrices 
   * 
   * @param v	First parameter of binary functor, destination matrix 
   * @param w	Second parameter of binary functor 
   * @param bf	 BinaryFunctor to apply
   * @param param Scalar parameter 
   */
  template<class V,class M,class I,class V2,class P>
	  void apply_binary_functor(dev_dense_matrix<V,M,I>& v, dev_dense_matrix<V2,M,I>& w, const BinaryFunctor& bf, const P& param){apply_binary_functor(v.vec(),w.vec(),bf,param);}
  /**
   * @copydoc apply_binary_functor(dev_dense_matrix<V,M,I>& v, dev_dense_matrix<V2,M,I>& w, const BinaryFunctor& bf, const P& param)
   */
  template<class V,class M,class I,class V2,class P>
	  void apply_binary_functor(host_dense_matrix<V,M,I>& v, host_dense_matrix<V2,M,I>& w, const BinaryFunctor& bf, const P& param){apply_binary_functor(v.vec(),w.vec(),bf,param);}

  /** 
   * @brief Apply pointwise binary functor with two scalar parameters to a pair of matrices 
   * 
   * @param v	First parameter of binary functor, destination matrix 
   * @param w	Second parameter of binary functor 
   * @param bf	 BinaryFunctor to apply
   * @param param First scalar parameter 
   * @param param2 Secont scalar parameter 
   *
   */
  template<class V,class M,class I,class V2,class P>
	  void apply_binary_functor(dev_dense_matrix<V,M,I>& v, dev_dense_matrix<V2,M,I>& w, const BinaryFunctor& bf, const P& param, const P& param2)
	  {apply_binary_functor(v.vec(),w.vec(),bf,param,param2);}
  /**
   * @copydoc void apply_binary_functor(dev_dense_matrix<V,M,I>& v, dev_dense_matrix<V2,M,I>& w, const BinaryFunctor& bf, const P& param, const P& param2)
   */
  template<class V,class M,class I,class V2,class P>
	  void apply_binary_functor(host_dense_matrix<V,M,I>& v, host_dense_matrix<V2,M,I>& w, const BinaryFunctor& bf, const P& param, const P& param2)
	  {apply_binary_functor(v.vec(),w.vec(),bf,param,param2);}

  // convenience wrappers

  /** 
   * @brief Copy one matrix into another. 
   * 
   * @param dst Destination matrix
   * @param src	Source matrix 
   * 
   * This is a convenience wrapper that applies the binary functor BF_COPY 
   */
  template<class V,class M, class I>
	  void copy(dev_dense_matrix<V,M,I>& dst, dev_dense_matrix<V,M,I>& src){ apply_binary_functor(dst.vec(),src.vec(),BF_COPY); }
  /**
   * @copydoc copy(dev_dense_matrix<V,M,I>& dst, dev_dense_matrix<V,M,I>& src)
   */
  template<class V,class M, class I>
	  void copy(host_dense_matrix<V,M,I>& dst, host_dense_matrix<V,M,I>& src){ apply_binary_functor(dst.vec(),src.vec(),BF_COPY); }

 /** @} */ // end of group functors_matrices


 /** @defgroup reductions_matrices Functors reducing a matrix to a scalar
  * @{
  */
  /** 
   * @brief Check whether a float matrix contains "Inf" or "-Inf"
   * 
   * @param v Target matrix 
   * 
   * @return true if v contains "Inf" or "-Inf", false otherwise 
   */
  template<class V,class M, class I> bool has_inf(dev_dense_matrix<V,M,I>& v){return has_inf(v.vec());}
  /** 
   * @brief Check whether a float matrix contains "NaN"
   * 
   * @param v Target matrix 
   * 
   * @return true if v contains "NaN", false otherwise 
   */
  template<class V,class M, class I> bool has_nan(dev_dense_matrix<V,M,I>& v){return has_nan(v.vec());}

  /** 
   * @brief Return the two-norm or Euclidean norm of a matrix 
   * 
   * @param v Target matrix
   * 
   * @return Two-norm of v 
   */
  template<class V,class M, class I> float norm2(dev_dense_matrix<V,M,I>& v){return norm2(v.vec());}

  /** 
   * @brief Return the one-norm or sum-norm of a matrix 
   * 
   * @param v Target matrix
   * 
   * @return one-norm of v 
   */
  template<class V,class M, class I> float norm1(dev_dense_matrix<V,M,I>& v){return norm1(v.vec());}
  /** 
   * @brief Return the minimum entry of a matrix 
   * 
   * @param v Target matrix
   * 
   * @return Minimum entry of v 
   */
  template<class V,class M, class I> float minimum(dev_dense_matrix<V,M,I>& v){return minimum(v.vec());}
  /** 
   * @brief Return the maximum entry of a matrix 
   * 
   * @param v Target matrix
   * 
   * @return Maximum entry of v 
   */
  template<class V,class M, class I> float maximum(dev_dense_matrix<V,M,I>& v){return maximum(v.vec());}
  /** 
   * @brief Return the mean of the entries of a matrix 
   * 
   * @param v Target matrix
   * 
   * @return Mean of entries of v 
   */
  template<class V,class M, class I> float mean(dev_dense_matrix<V,M,I>& v) {return mean(v.vec());}
  /** 
   * @brief Return the variation of the entries of a matrix 
   * 
   * @param v Target matrix
   * 
   * @return Variation of entries of v 
   */
  template<class V,class M, class I> float var(dev_dense_matrix<V,M,I>& v)  {return var(v.vec());}

  /**
   * @copydoc has_inf(dev_dense_matrix<V,M,I>& v)
   */
  template<class V,class M, class I> bool has_inf(host_dense_matrix<V,M,I>& v){return has_inf(v.vec());}
  /**
   * @copydoc has_nan(dev_dense_matrix<V,M,I>& v)
   */
  template<class V,class M, class I> bool has_nan(host_dense_matrix<V,M,I>& v){return has_nan(v.vec());}
  /**
   * @copydoc norm2(dev_dense_matrix<V,M,I>& v)
   */
  template<class V,class M, class I> float norm2(host_dense_matrix<V,M,I>& v){return norm2(v.vec());}
  /**
   * @copydoc norm1(dev_dense_matrix<V,M,I>& v)
   */
  template<class V,class M, class I> float norm1(host_dense_matrix<V,M,I>& v){return norm1(v.vec());}
  /**
   * @copydoc minimum(dev_dense_matrix<V,M,I>& v)
   */
  template<class V,class M, class I> float minimum(host_dense_matrix<V,M,I>& v){return minimum(v.vec());}
  /**
   * @copydoc minimum(dev_dense_matrix<V,M,I>& v)
   */
  template<class V,class M, class I> float maximum(host_dense_matrix<V,M,I>& v){return maximum(v.vec());}
  /**
   * @copydoc mean(dev_dense_matrix<V,M,I>& v)
   */
  template<class V,class M, class I> float mean(host_dense_matrix<V,M,I>& v) {return mean(v.vec());}
  /**
   * @copydoc var(dev_dense_matrix<V,M,I>& v)
   */
  template<class V,class M, class I> float var(host_dense_matrix<V,M,I>& v)  {return var(v.vec());}


 /** @} */ // end of group reductions_matrices


 /** @defgroup reductions Reductions from matrix to row or column
  * @{
  */

  /** 
   * @brief Reduce functor to reduce a matrix to a row or column
   * 
   * 	- RF_ADD adds columns/rows
   * 	- RF_ADD_SQUARED adds squared entries of columns/rows
   * 	- RF_MAX uses maximum in colum (when reducing to row) or row (when reducing to column)
   * 	- RF_MIN uses minimum in colum (when reducing to row) or row (when reducing to column)
   */
  enum reduce_functor{
	  RF_ADD,
	  RF_ADD_SQUARED,
	  RF_MAX,
	  RF_MIN,
  };

  /** 
   * @brief Reduce a matrix to one column using specified reduce functor (or add them up by default)
   * 
   * @param dst Destination vector, dst.size = src.h()
   * @param src Source matrix
   * @param rf	Reduce functor 
   * @param factNew Scalar factor for result of reduce functor 
   * @param factOld Scalar factor for former entry of dst 
   *	 
   *	 Calculates
   *	 dst= factOld * dst + factNew * rf(src)
   *	 By default, the reduce functor is RF_ADD so that rf(src) is the sum over all columns of src.
   */
  template<class __matrix_type, class __vector_type> 
	  void reduce_to_col(__vector_type& dst, const __matrix_type& src, reduce_functor rf=RF_ADD, const typename __matrix_type::value_type& factNew=1.f, const typename __matrix_type::value_type& factOld=0.f);

  /** 
   * @brief Reduce a matrix to one row using specified reduce functor (or add them up by default)
   * 
   * @param dst Destination vector, dst.size = src.w()
   * @param src Source matrix
   * @param rf	Reduce functor 
   * @param factNew Scalar factor for result of reduce functor 
   * @param factOld Scalar factor for former entry of dst 
   *	 
   *	 Calculates
   *	 dst= factOld * dst + factNew * rf(src)
   *	 By default, the reduce functor is RF_ADD so that rf(src) is the sum over all rows of src.
   */
  template<class __matrix_type, class __vector_type> 
	  void reduce_to_row(__vector_type& dst, const __matrix_type& src, reduce_functor rf=RF_ADD, const typename __matrix_type::value_type& factNew=1.f, const typename __matrix_type::value_type& factOld=0.f);

  /** 
   * @brief Write the index of the maximum for each column of a matrix into a vector
   * 
   * @param dst Destination vector, dst.size = src.h()
   * @param src Source matrix 
   */
  template<class __matrix_type, class __vector_type>
	  void argmax_to_row(__vector_type& dst, const __matrix_type& src);

  /** 
   * @brief Write the index of the maximum for each row of a matrix into a vector
   * 
   * @param dst Destination vector, dst.size = src.w() 
   * @param src Source matrix
   * 
   */
  template<class __matrix_type, class __vector_type>
	  void argmax_to_column(__vector_type& dst, const __matrix_type& src);

 /** @} */ // end of group reductions


  /***************************************************
  * Get view on parts of matrix
  * *************************************************/
  /** 
   * @brief Generate a view to a block inside an existing matrix
   * 
   * @param matrix Matrix to generate a view from 
   * @param start_rows 	First row in block 
   * @param num_rows	Number of rows in block 
   * @param start_cols	First column in block 
   * @param num_cols	Number of columns in block 
   * 
   * @return View of specified block inside matrix 
   *
   * 	Returns a matrix of size num_rows x num_cols that is a view to the entries of matrix,
   * 	starting at entry start_rows,start_cols to entry start_rows+num_rows,start_cols+num_cols.
   * 	For row major matrices, this only works with start_cols=0 and num_cols=matrix.w().
   * 	For column major matrices, this only works with start_rows=0 and num_rows=matrix.h().
   */
  template<class __matrix_type>
	  __matrix_type* blockview(__matrix_type & matrix, typename __matrix_type::index_type start_rows, typename __matrix_type::index_type num_rows ,typename __matrix_type::index_type start_cols, typename __matrix_type::index_type num_cols);


  /***************************************************
   * BLAS3 stuff
   ***************************************************/
 /** @defgroup blas3 BLAS3
  * @{
  */
  /** 
   * @brief Matrix multiplication and other BLAS3 functionality.
   * 
   * @param dst Destination matrix 
   * @param A First matrix for product 
   * @param B Second matrix for product 
   * @param transA Whether to transpose A before calculating the matix product. Possible values 'n' for "do Not transpose" and 't' for "Transpose". 
   * @param transB Whether to transpose B befor calculating the matrix product. Possible values 'n' for "do Not transpose" and 't' for "Transpose". 
   * @param factAB Scalar factor to multiply the product of A and B with. 
   * @param factC Scalar factor to multiply former entries of C with.
   * 
   * Calculates C = factC * C + factAB * transA(A)*transB(B)
   * Here transA(A) is the transpose of A if transA = 't' and transA(A) is A if transA = 'n'.
   * The analogue is true for transB(B).
   * In the above transA(A)*transB(B) is the matrix product and all other operations are pointwise.
   * This is a thin wrapper of CUBLAS.
   */
  template<class __matrix_type, class __matrix_type2, class __matrix_type3>
	  void prod(__matrix_type& dst, __matrix_type2& A, __matrix_type3& B, char transA='n', char transB='n', const float& factAB=1.f, const float& factC=0.f);

  /** @} */ // end group blas3

  /***************************************************
   * BLAS2 stuff
   ***************************************************/
 /** @defgroup blas2 BLAS2
  * @{
  */
  /** 
   * @brief Calculates product of a sparse matrix and a vector. 
   * 
   * @param dst	 Destination vector
   * @param A	 Sparse matrix 
   * @param v	 Input vector 
   * @param transA Wether to transpose A before calculating the matrix product
   * @param factAv Scalar factor for product of A and v 
   * @param factC	Scalar factor for former value of C.
   *
   * 	Calculates:
   *	dst = factC * dst + factAv*(transA(A)*v)
   *
   *	Here transA(A) is the transpose of A if transA = 't' and transA(A) is A if transA = 'n'.
   *	transA(A)*v is the matrix-vector product and all other operations are pointwise.
   */
  template<class __matrix_type, class __vector_type>
	  void spmv(__vector_type& dst, __matrix_type& A, __vector_type& v, char transA='n', const float& factAv=1.f, const float& factC=0.f);
  
  /** 
   * @brief Add a vector to each column of a matrix A.
   * 
   * @param A Destination matrix 
   * @param v Vector, v.size()=A.h() 
   * 
   */
  template<class __matrix_type, class __vector_type>
	  void matrix_plus_col(__matrix_type& A, const __vector_type& v);

  /** 
   * @brief Multiply each column of a matrix A pointwise with a vector v.
   * 
   * @param A Destination matrix 
   * @param v Vector, v.size()=A.h() 
   * 
   */
  template<class __matrix_type, class __vector_type>
	  void matrix_times_col(__matrix_type& A, const __vector_type& v);

  /** 
   * @brief Devide each column of a matrix A pointwise by a vector v.
   * 
   * @param A Destination matrix 
   * @param v Vector, v.size()=A.h() 
   * 
   */
  template<class __matrix_type, class __vector_type>
	  void matrix_divide_col(__matrix_type& A, const __vector_type& v);

  /** 
   * @brief Transpose a matrix
   * 
   * @param dst Destination matrix 
   * @param src Source matrix 
   * 
   */
  template<class V,class M, class I> void transpose(dev_dense_matrix<V,M,I>&  dst, dev_dense_matrix<V,M,I>&  src);
  /**
   * @copydoc transpose(dev_dense_matrix<V,M,I>&  dst, dev_dense_matrix<V,M,I>&  src)
   */
  template<class V,class M, class I> void transpose(host_dense_matrix<V,M,I>& dst, host_dense_matrix<V,M,I>& src);

  /** @} */ // end group blas2
} // cuv

#endif
