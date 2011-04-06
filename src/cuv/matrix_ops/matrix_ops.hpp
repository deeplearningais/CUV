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





#ifndef __MATRIX_OPS_HPP__
#define __MATRIX_OPS_HPP__

#include <cuv/basics/dense_matrix.hpp>
#include <cuv/basics/dia_matrix.hpp>

namespace cuv{

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
	  RF_ARGMAX,
	  RF_ARGMIN,
	  RF_MIN,
	  RF_MULT,
	  RF_LOGADDEXP,
	  RF_ADDEXP,
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
  template<class __value_type, class __value_type2, class __memory_space_type, class __memory_layout_type>
	  void reduce_to_col(tensor<__value_type, __memory_space_type>& dst, const dense_matrix<__value_type2, __memory_space_type, __memory_layout_type>& src, reduce_functor rf=RF_ADD, const __value_type2& factNew=1.f, const __value_type2& factOld=0.f);

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
  template<class __value_type, class __value_type2, class __memory_space_type, class __memory_layout_type>
	  void reduce_to_row(tensor<__value_type, __memory_space_type>& dst, const dense_matrix<__value_type2, __memory_space_type, __memory_layout_type>& src, reduce_functor rf=RF_ADD, const __value_type2& factNew=1.f, const __value_type2& factOld=0.f);

  /** 
   * @brief Write the index of the maximum for each column of a matrix into a vector
   * 
   * @param dst Destination vector, dst.size = src.h()
   * @param src Source matrix 
   */
  template<class __value_type, class __memory_space_type, class __memory_layout_type>
	  void argmax_to_row(tensor<__value_type, __memory_space_type>& dst, const dense_matrix<__value_type, __memory_space_type, __memory_layout_type>& src);

  /** 
   * @brief Write the index of the maximum for each row of a matrix into a vector
   * 
   * @param dst Destination vector, dst.size = src.w() 
   * @param src Source matrix
   * 
   */
  template<class __value_type, class __memory_space_type, class __memory_layout_type>
	  void argmax_to_column(tensor<__value_type, __memory_space_type>& dst, const dense_matrix<__value_type, __memory_space_type, __memory_layout_type>& src);

 /** @} */ // end of group reductions

  /** 
   * @brief Bit-Flip a row of a column-major matrix
   * 
   * @param matrix Matrix to apply functor on
   * @param row	   row to flip
   * 
   * changes the matrix such that its m-th row is now (1-original mth row)
   *
   */
  template<class __value_type, class __memory_layout, class __memory_space_type, class __index_type>
	  void bitflip(
	  dense_matrix<__value_type,__memory_layout,__memory_space_type,__index_type> & matrix,
			  __index_type row);


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
  template<class __value_type, class __memory_layout, class __memory_space_type, class __index_type>
	  dense_matrix<__value_type,__memory_layout,__memory_space_type,__index_type>* blockview(
	  dense_matrix<__value_type,__memory_layout,__memory_space_type,__index_type> & matrix,
			  __index_type start_rows,
			  __index_type num_rows ,
			  __index_type start_cols,
			  __index_type num_cols);


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
  template<class __matrix_type, class __matrix_type1, class __matrix_type2>
	  void prod(__matrix_type& C, __matrix_type1& A, __matrix_type2& B, char transA='n', char transB='n', const float& factAB=1.f, const float& factC=0.f);

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
  template<class __value_type, class __memory_space_type>
	  void spmv(tensor<__value_type, __memory_space_type>& dst, dia_matrix<__value_type, __memory_space_type>& A, tensor<__value_type, __memory_space_type>& v, char transA='n', const float& factAv=1.f, const float& factC=0.f);
  
  /** 
   * @brief Add a vector to each column of a matrix A.
   * 
   * @param A Destination matrix 
   * @param v Vector, v.size()=A.h() 
   * 
   */
  template<class __value_type, class __memory_space_type, class __memory_layout_type>
	  void matrix_plus_col(dense_matrix<__value_type, __memory_space_type, __memory_layout_type>& A, const tensor<__value_type, __memory_space_type>& v);

  /** 
   * @brief Multiply each column of a matrix A pointwise with a vector v.
   * 
   * @param A Destination matrix 
   * @param v Vector, v.size()=A.h() 
   * 
   */
  template<class __value_type, class __memory_space_type, class __memory_layout_type>
	  void matrix_times_col(dense_matrix<__value_type, __memory_space_type, __memory_layout_type>& A, const tensor<__value_type, __memory_space_type>& v);

  /** 
   * @brief Devide each column of a matrix A pointwise by a vector v.
   * 
   * @param A Destination matrix 
   * @param v Vector, v.size()=A.h() 
   * 
   */
  template<class __value_type, class __memory_space_type, class __memory_layout_type>
	  void matrix_divide_col(dense_matrix<__value_type, __memory_space_type, __memory_layout_type>& A, const tensor<__value_type, __memory_space_type>& v);

  /** 
   * @brief Add a vector to each row of a matrix A.
   * 
   * @param A Destination matrix 
   * @param v Vector, v.size()=A.h() 
   * 
   */
  template<class __value_type, class __memory_space_type, class __memory_layout_type>
	  void matrix_plus_row(dense_matrix<__value_type, __memory_space_type, __memory_layout_type>& A, const tensor<__value_type, __memory_space_type>& v);

  /** 
   * @brief Multiply each row of a matrix A pointwise with a vector v.
   * 
   * @param A Destination matrix 
   * @param v Vector, v.size()=A.h() 
   * 
   */
  template<class __value_type, class __memory_space_type, class __memory_layout_type>
	  void matrix_times_row(dense_matrix<__value_type, __memory_space_type, __memory_layout_type>& A, const tensor<__value_type, __memory_space_type>& v);

  /** 
   * @brief Devide each row of a matrix A pointwise by a vector v.
   * 
   * @param A Destination matrix 
   * @param v Vector, v.size()=A.h() 
   * 
   */
  template<class __value_type, class __memory_space_type, class __memory_layout_type>
	  void matrix_divide_row(dense_matrix<__value_type, __memory_space_type, __memory_layout_type>& A, const tensor<__value_type, __memory_space_type>& v);

  /** 
   * @brief Transpose a matrix
   * 
   * @param dst Destination matrix 
   * @param src Source matrix 
   * 
   */
  template<class __matrix_type > void transpose(__matrix_type&  dst, const __matrix_type&  src);

  /** 
   * @brief Transpose a matrix by creating a view with different storage
   * 
   * @param dst Destination matrix 
   * @param src Source matrix 
   *
   * Creates a row major view of a column major matrix or a column major view of a row major matrix.
   * Does not actually modify the content of the memory.
   * 
   */
  template<class V, class T, class I>
  cuv::dense_matrix<V,row_major,T,I>* transposed_view(cuv::dense_matrix<V,column_major,T,I>&  src);
  template<class V, class T, class I>
  cuv::dense_matrix<V,column_major,T,I>* transposed_view(cuv::dense_matrix<V,row_major,T,I>&  src);

  /** @} */ // end group blas2
} // cuv

 /* 
  * operator overloading for arithmetic operations on matrices
  */
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I> 
    //operator- (const cuv::dense_matrix<V, M, T, I> & v){
          //cuv::dense_matrix<V, M, T, I> temp = v; 	
        //temp.vec()*= (V)-1.0;
        //return temp;
  //}
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I> 
    //operator+ (const cuv::dense_matrix<V, M, T, I> & v, const V & p){
          //cuv::dense_matrix<V, M, T, I> temp = v; 	
        //temp.vec()+= p;
        //return temp;
  //}
  
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I> 
    //operator- (const cuv::dense_matrix<V, M, T, I> & v, const V & p){
          //cuv::dense_matrix<V, M, T, I> temp = v; 	
        //temp.vec()-= p;
        //return temp;
  //}
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I> 
    //operator* (const cuv::dense_matrix<V, M, T, I> & v, const V & p){
          //cuv::dense_matrix<V, M, T, I> temp = v; 	
        //temp.vec()*= p;
        //return temp;
  //}
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I> 
    //operator/ (const cuv::dense_matrix<V, M, T, I> & v, const V & p){
          //cuv::dense_matrix<V, M, T, I> temp = v; 	
        //temp.vec()/= p;
        //return temp;
  //}
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I> 
    //operator+ (const cuv::dense_matrix<V, M, T, I> & v1, const cuv::dense_matrix<V, M, T, I> & v2){
          //cuv::dense_matrix<V, M, T, I> temp = v1; 	
        //temp.vec()+= v2.vec();
        //return temp;
  //}
  
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I> 
    //operator- (const cuv::dense_matrix<V, M, T, I> & v1, const cuv::dense_matrix<V, M, T, I> & v2){
          //cuv::dense_matrix<V, M, T, I> temp = v1;	
        //temp.vec()-= v2.vec();
        //return temp;
  //}
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I> 
    //operator* (const cuv::dense_matrix<V, M, T, I> & v1, const cuv::dense_matrix<V, M, T, I> & v2){
          //cuv::dense_matrix<V, M, T, I> temp = v1; 	
        //temp.vec()*= v2.vec();
        //return temp;
  //}
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I> 
    //operator/ (const cuv::dense_matrix<V, M, T, I> & v1, const cuv::dense_matrix<V, M, T, I> & v2){
          //cuv::dense_matrix<V, M, T, I> temp = v1; 	
        //temp.vec()/= v2.vec();
        //return temp;
  //}
  
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I>& 
    //operator-=(cuv::dense_matrix<V, M, T, I>& v1, const cuv::dense_matrix<V, M, T, I>& v2){
          //cuv::apply_binary_functor(v1.vec(), v2.vec(), cuv::BF_SUBTRACT);
          //return v1;
  //}
  
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I>& 
    //operator*=(cuv::dense_matrix<V, M, T, I>& v1, const cuv::dense_matrix<V, M, T, I>& v2){
          //cuv::apply_binary_functor(v1.vec(), v2.vec(), cuv::BF_MULT);
          //return v1;
  //}
  
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I>& 
    //operator/=(cuv::dense_matrix<V, M, T, I>& v1, const cuv::dense_matrix<V, M, T, I>& v2){
          //cuv::apply_binary_functor(v1.vec(), v2.vec(), cuv::BF_DIV);
          //return v1;
  //}
  
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I>& 
    //operator+=(cuv::dense_matrix<V, M, T, I>& v1, const cuv::dense_matrix<V, M, T, I>& v2){
          //cuv::apply_binary_functor(v1.vec(), v2.vec(), cuv::BF_ADD);
          //return v1;
  //}
  
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I>& 
    //operator-=(cuv::dense_matrix<V, M, T, I>& v, const V& p){
          //cuv::apply_scalar_functor(v.vec(), cuv::SF_SUBTRACT, p);
          //return v;
  //}
  
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I>& 
    //operator*=(cuv::dense_matrix<V, M, T, I>& v, const V& p){
          //cuv::apply_scalar_functor(v.vec(), cuv::SF_MULT, p);
          //return v;
  //}
  
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I>& 
    //operator/=(cuv::dense_matrix<V, M, T, I>& v, const V& p){
          //cuv::apply_scalar_functor(v.vec(), cuv::SF_DIV, p);
          //return v;
  //}
  
  //template<class V, class M, class T, class I>
    //cuv::dense_matrix<V, M, T, I>& 
    //operator+=(cuv::dense_matrix<V, M, T, I>& v, const V& p){
          //cuv::apply_scalar_functor(v.vec(), cuv::SF_ADD, p);
          //return v;
  //}

  //template<class V, class M, class T, class I, class V2>
    //void fill(cuv::dense_matrix<V, M, T, I>& m, V2 v){
	   //cuvAssert(false); //not implemented!
    //}
  
#endif
