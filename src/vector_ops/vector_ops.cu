#include <functional>

#include <dev_vector.hpp>
#include <host_vector.hpp>

#include "vector_ops.hpp"

#define sgn(a) (copysign(1.f,a))


template<class T>
struct uf_exp      {  __global__    T operator()(T t)      { return __expf(t); } };
template<class T>
struct uf_exact_exp{  __global__ __host__ T operator()(T t){ return exp(t);    } };
template<class T>
struct uf_log{  __global__ __host__ T operator()(T t)      { return log(t);    } };
template<class T>
struct uf_sign{  __global__         T operator()(T t)      { return sgn(t);    } };
template<class T>
struct uf_sigm{  __global__         T operator()(T t)      { return ((T)1)/(((T)1)+__expf(-t));    } };
template<class T>
struct uf_exact_sigm{  __global__  __host__ T operator()(T t)      { return ((T)1)/(((T)1)+exp(-t));    } };
template<class T>
struct uf_dsigm{  __global__ __host__       T operator()(T t)      { return t * (((T)1)-t); } };
template<class T>
struct uf_tanh{  __global__  __host__       T operator()(T t)      { return tanh(t); } };
template<class T>
struct uf_dtanh{  __global__  __host__      T operator()(T t)      { return ((T)1) - (t*t); } };
template<class T>
struct uf_square{  __global__  __host__     T operator()(T t)      { return t*t; } };
template<class T>
struct uf_sublin{  __global__  __host__     T operator()(T t)      { return ((T)1)-t; } };
template<class T>
struct uf_energ{  __global__  __host__      T operator()(T t)      { return -log(t); } };
template<class T>
struct uf_inv{  __global__  __host__        T operator()(T t)      { return ((T)1)/(t+((T)0.00000001)); } };
template<class T>
struct uf_sqrt{  __global__  __host__       T operator()(T t)      { return sqrt(t); } };

template<class T, class binary_functor>
struct uf_base_op{
  T x;
	uf_base_op(_x):x(_x){};
  T operator()(T t){ return binary_functor(t,x); }
};

template<class unary_functor, class value_type, class index_type>
__global__
void unary_functor_kernel(value_type* dst, value_type* src, index_type n, unary_functor uf){
	const unsigned int idx = __mul24(blockIdx.x , blockDim.x) + threadIdx.x;
	const unsigned int off = __mul24(blockDim.x , gridDim.x);
	for (unsigned int i = idx; i < numElements; i += off)
		target[i] = uf(src[i]);
}

template<class unary_functor, class value_type, class index_type>
void launch_unary_kernel(
   dev_vector<value_type, index_type>& dst,
   dev_vector<value_type, index_type>& src, 
	 unary_functor uf){
	 cuvAssert(dst.ptr());
	 cuvAssert(src.ptr());
	 cuvAssert(dst.size() == src.size());
	 // TODO: determine blocks, threads
   unary_functor_kernel<<blocks,threads>>>(dst.ptr(),src.ptr(),dst.size(),uf);
}

template<class unary_functor, class value_type, class index_type>
void launch_unary_kernel(
   host_vector<value_type, index_type>& dst,
   host_vector<value_type, index_type>& src, 
	 unary_functor uf){
	 cuvAssert(src.ptr());
	 cuvAssert(dst.ptr());
	 cuvAssert(dst.size() == src.size());
	 for(int i=0;i<dst.size();i++)
	   dst[i] = uf(src[i]);
}

namespace cuv{

template<class __vector_type>
apply_scalar_functor(__vector_type& v, const ScalarFunctor& sf){
  apply_scalar_functor_impl<__vector_type>::apply(v,sf);
}
template<class __vector_type, class __value_type>
apply_scalar_functor(__vector_type& v, const ScalarFunctor& sf, const __value_type& param){
  apply_scalar_functor_impl<__vector_type>::apply(v,sf,param);
}

template<class __vector_type>
struct apply_scalar_functor_impl{

	template<class value_type>
	apply(__vector_type& v, const ScalarFunctor& sf, const value_type& param){
	  switch(sf){
		  SF_ADD:  launch_unary_kernel(v,v,uf_base_op<__value_type, std::plus<__value_type> >(param)); break;
		  SF_MULT: launch_unary_kernel(v,v,uf_base_op<__value_type, std::multiplies<__value_type> >(param)); break;
		  SF_DIV:  launch_unary_kernel(v,v,uf_base_op<__value_type, std::divides<__value_type> >(param)); break;
		  SF_SUB:  launch_unary_kernel(v,v,uf_base_op<__value_type, std::minus<__value_type> >(param)); break;
		}
	}

	apply(__vector_type& v, const ScalarFunctor& sf){
	  switch(sf){
			SF_EXP:        launch_unary_kernel(v,v, uf_exp<__value_type>()); break;
			SF_EXACT_EXP:  launch_unary_kernel(v,v, uf_exact_exp<__value_type>()); break;
			SF_LOG:        launch_unary_kernel(v,v, uf_log<__value_type>()); break;
			SF_SIGN:       launch_unary_kernel(v,v, uf_sign<__value_type>()); break;
			SF_SIGM:       launch_unary_kernel(v,v, uf_sigm<__value_type>()); break;
			SF_DSIGM:      launch_unary_kernel(v,v, uf_dsigm<__value_type>()); break;
			SF_TANH:       launch_unary_kernel(v,v, uf_tanh<__value_type>()); break;
			SF_DTANH:      launch_unary_kernel(v,v, uf_dtanh<__value_type>()); break;
			SF_SQUARE:     launch_unary_kernel(v,v, uf_square<__value_type>()); break;
			SF_SUBLIN:     launch_unary_kernel(v,v, uf_sublin<__value_type>()); break;
			SF_ENERG:      launch_unary_kernel(v,v, uf_energ<__value_type>()); break;
			SF_INV:        launch_unary_kernel(v,v, uf_inv<__value_type>()); break;
			SF_SQRT:       launch_unary_kernel(v,v, uf_sqrt<__value_type>()); break;
			default:
			 cuvAssert(false);
		}
	}
};


} // cuv
