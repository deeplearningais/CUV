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
  
  /*
   * Pointwise Null-ary Functor
   * v = sf()
   */
  template<class V,class M,class I> 
	  void apply_0ary_functor(dev_dense_matrix<V,M,I>& m, const NullaryFunctor& sf){ apply_0ary_functor(m.vec(),sf); }
  template<class V,class M,class I,class P> 
	  void apply_0ary_functor(dev_dense_matrix<V,M,I>& m, const NullaryFunctor& sf, const P& param){apply_0ary_functor(m.vec(),sf,param);}
  template<class V,class M,class I> 
	  void apply_0ary_functor(host_dense_matrix<V,M,I>& m, const NullaryFunctor& sf){ apply_0ary_functor(m.vec(),sf); }
  template<class V,class M,class I,class P> 
	  void apply_0ary_functor(host_dense_matrix<V,M,I>& m, const NullaryFunctor& sf, const P& param){apply_0ary_functor(m.vec(),sf,param);}

  /*
   * Pointwise Unary Functor
   * v = sf(v)
   */
  template<class V,class M,class I> 
	  void apply_scalar_functor(dev_dense_matrix<V,M,I>& m, const ScalarFunctor& sf){apply_scalar_functor(m.vec(),sf);}
  template<class V,class M,class I> 
	  void apply_scalar_functor(host_dense_matrix<V,M,I>& m, const ScalarFunctor& sf){apply_scalar_functor(m.vec(),sf);}

  /*
   * Pointwise Unary Functor with scalar parameter
   * v = sf(v, param)
   */
  template<class V,class M,class I,class P> 
	  void apply_scalar_functor(dev_dense_matrix<V,M,I>& m, const ScalarFunctor& sf, const P& p) {apply_scalar_functor(m.vec(),sf,p);};
  template<class V,class M,class I,class P> 
	  void apply_scalar_functor(host_dense_matrix<V,M,I>& m, const ScalarFunctor& sf, const P& p) {apply_scalar_functor(m.vec(),sf,p);};

  /*
   * Pointwise Unary Functor with two scalar parameters
   * v = sf(v, param, param2)
   */
  template<class V,class M,class I,class P>
	  void apply_scalar_functor(dev_dense_matrix<V,M,I>& m, const ScalarFunctor& sf, const P& p, const P& p2) {apply_scalar_functor(m.vec(),sf,p,p2);};
  template<class V,class M,class I,class P>
	  void apply_scalar_functor(host_dense_matrix<V,M,I>& m, const ScalarFunctor& sf, const P& p, const P& p2) {apply_scalar_functor(m.vec(),sf,p,p2);};

  /*
   * Pointwise Binary Functor
   * v = bf(v,w)
   */
  template<class V,class M,class I,class V2> 
	  void apply_binary_functor(dev_dense_matrix<V,M,I>& v, dev_dense_matrix<V2,M,I>& w, const BinaryFunctor& bf){apply_binary_functor(v.vec(),w.vec(),bf);}
  template<class V,class M,class I,class V2> 
	  void apply_binary_functor(host_dense_matrix<V,M,I>& v, host_dense_matrix<V2,M,I>& w, const BinaryFunctor& bf){apply_binary_functor(v.vec(),w.vec(),bf);}

  /*
   * Pointwise Binary Functor
   * v = bf(v,w,param)
   */
  template<class V,class M,class I,class V2,class P>
	  void apply_binary_functor(dev_dense_matrix<V,M,I>& v, dev_dense_matrix<V2,M,I>& w, const BinaryFunctor& bf, const P& param){apply_binary_functor(v.vec(),w.vec(),bf,param);}
  template<class V,class M,class I,class V2,class P>
	  void apply_binary_functor(host_dense_matrix<V,M,I>& v, host_dense_matrix<V2,M,I>& w, const BinaryFunctor& bf, const P& param){apply_binary_functor(v.vec(),w.vec(),bf,param);}

  /*
   * Pointwise Binary Functor
   * v = bf(v,w,param,param2)
   */
  template<class V,class M,class I,class V2,class P>
	  void apply_binary_functor(dev_dense_matrix<V,M,I>& v, dev_dense_matrix<V2,M,I>& w, const BinaryFunctor& bf, const P& param, const P& param2)
	  {apply_binary_functor(v.vec(),w.vec(),bf,param,param2);}
  template<class V,class M,class I,class V2,class P>
	  void apply_binary_functor(host_dense_matrix<V,M,I>& v, host_dense_matrix<V2,M,I>& w, const BinaryFunctor& bf, const P& param, const P& param2)
	  {apply_binary_functor(v.vec(),w.vec(),bf,param,param2);}

  // convenience wrappers
  template<class V,class M, class I>
	  void copy(dev_dense_matrix<V,M,I>& dst, dev_dense_matrix<V,M,I>& src){ apply_binary_functor(dst.vec(),src.vec(),BF_COPY); }
  template<class V,class M, class I>
	  void copy(host_dense_matrix<V,M,I>& dst, host_dense_matrix<V,M,I>& src){ apply_binary_functor(dst.vec(),src.vec(),BF_COPY); }

  /*
   * transpose
   *
   */

  template<class V,class M, class I> void transpose(dev_dense_matrix<V,M,I>&  dst, dev_dense_matrix<V,M,I>&  src);
  template<class V,class M, class I> void transpose(host_dense_matrix<V,M,I>& dst, host_dense_matrix<V,M,I>& src);

  /*
   * reductions
   *
   */
  template<class V,class M, class I> bool has_inf(dev_dense_matrix<V,M,I>& v){return has_inf(v.vec());}
  template<class V,class M, class I> bool has_nan(dev_dense_matrix<V,M,I>& v){return has_nan(v.vec());}
  template<class V,class M, class I> float norm2(dev_dense_matrix<V,M,I>& v){return norm2(v.vec());}
  template<class V,class M, class I> float norm1(dev_dense_matrix<V,M,I>& v){return norm1(v.vec());}
  template<class V,class M, class I> float minimum(dev_dense_matrix<V,M,I>& v){return minimum(v.vec());}
  template<class V,class M, class I> float maximum(dev_dense_matrix<V,M,I>& v){return maximum(v.vec());}
  template<class V,class M, class I> float mean(dev_dense_matrix<V,M,I>& v) {return mean(v.vec());}
  template<class V,class M, class I> float var(dev_dense_matrix<V,M,I>& v)  {return var(v.vec());}

  template<class V,class M, class I> bool has_inf(host_dense_matrix<V,M,I>& v){return has_inf(v.vec());}
  template<class V,class M, class I> bool has_nan(host_dense_matrix<V,M,I>& v){return has_nan(v.vec());}
  template<class V,class M, class I> float norm2(host_dense_matrix<V,M,I>& v){return norm2(v.vec());}
  template<class V,class M, class I> float norm1(host_dense_matrix<V,M,I>& v){return norm1(v.vec());}
  template<class V,class M, class I> float minimum(host_dense_matrix<V,M,I>& v){return minimum(v.vec());}
  template<class V,class M, class I> float maximum(host_dense_matrix<V,M,I>& v){return maximum(v.vec());}
  template<class V,class M, class I> float mean(host_dense_matrix<V,M,I>& v) {return mean(v.vec());}
  template<class V,class M, class I> float var(host_dense_matrix<V,M,I>& v)  {return var(v.vec());}

  enum reduce_functor{
	  RF_ADD,
	  RF_ADD_SQUARED,
	  RF_MAX,
	  RF_MIN,
  };

  /// sum all columns of a matrix to get one sum-column
  template<class __matrix_type, class __vector_type> 
	  void reduce_to_col(__vector_type&, const __matrix_type&, reduce_functor rf=RF_ADD, const typename __matrix_type::value_type& factNew=1.f, const typename __matrix_type::value_type& factOld=0.f);

  /// sum all rows of a matrix to get one sum-row
  template<class __matrix_type, class __vector_type> 
	  void reduce_to_row(__vector_type&, const __matrix_type&, const typename __matrix_type::value_type& factNew=1.f, const typename __matrix_type::value_type& factOld=0.f);

  /// write the index of the maximum for each column of a matrix into a vector
  template<class __matrix_type, class __vector_type>
	  void argmax_to_row(__vector_type&, const __matrix_type&);

  /// write the index of the maximum for each row of a matrix into a vector
  template<class __matrix_type, class __vector_type>
	  void argmax_to_column(__vector_type&, const __matrix_type&);

  // end of wrappers for vector ops
 
  /***************************************************
  * Get view on parts of matrix
  * *************************************************/
  template<class __matrix_type>
	  __matrix_type* blockview(__matrix_type & matrix, typename __matrix_type::index_type start_rows, typename __matrix_type::index_type num_rows ,typename __matrix_type::index_type start_cols, typename __matrix_type::index_type num_cols);




  /***************************************************
   * BLAS3 stuff
   ***************************************************/
  template<class __matrix_type, class __matrix_type2, class __matrix_type3>
	  void prod(__matrix_type& dst, __matrix_type2& A, __matrix_type3& B, char transA='n', char transB='n', const float& factAB=1.f, const float& factC=0.f);

  /***************************************************
   * BLAS2 stuff
   ***************************************************/
  // dst = factC * dst + factAv*(A*v)
  // where 
  // factC, factAv:  scalar
  // A            :  sparse matrix
  // v            :  vector
  template<class __matrix_type, class __vector_type>
	  void spmv(__vector_type& dst, __matrix_type& A, __vector_type& v, char transA='n', const float& factAv=1.f, const float& factC=0.f);
  
  template<class __matrix_type, class __vector_type>
	  void matrix_plus_col(__matrix_type& A, const __vector_type& v);

  template<class __matrix_type, class __vector_type>
	  void matrix_times_col(__matrix_type& A, const __vector_type& v);

  template<class __matrix_type, class __vector_type>
	  void matrix_divide_col(__matrix_type& A, const __vector_type& v);
} // cuv

#endif
