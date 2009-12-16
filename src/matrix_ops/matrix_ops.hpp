#ifndef __MATRIX_OPS_HPP__
#define __MATRIX_OPS_HPP__

#include <vector_ops.hpp>
#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>

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
   * reductions
   *
   */
  template<class V,class M, class I> float norm2(dev_dense_matrix<V,M,I>& v){return norm2(v.vec());}
  template<class V,class M, class I> float norm1(dev_dense_matrix<V,M,I>& v){return norm1(v.vec());}
  template<class V,class M, class I> float mean(dev_dense_matrix<V,M,I>& v) {return mean(v.vec());}
  template<class V,class M, class I> float var(dev_dense_matrix<V,M,I>& v)  {return var(v.vec());}

  template<class V,class M, class I> float norm2(host_dense_matrix<V,M,I>& v){return norm2(v.vec());}
  template<class V,class M, class I> float norm1(host_dense_matrix<V,M,I>& v){return norm1(v.vec());}
  template<class V,class M, class I> float mean(host_dense_matrix<V,M,I>& v) {return mean(v.vec());}
  template<class V,class M, class I> float var(host_dense_matrix<V,M,I>& v)  {return var(v.vec());}


  /*
   * BLAS3 stuff
   *
   */
  template<class __matrix_type>
	  void prod(__matrix_type& dst, __matrix_type& A, __matrix_type& B, char transA='n', char transB='n', const float& factAB=1.f, const float& factC=0.f);


} // cuv

#endif
