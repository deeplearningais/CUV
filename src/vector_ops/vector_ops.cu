#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>

#include <cuv_general.hpp>

#include <dev_vector.hpp>
#include <host_vector.hpp>

#include "vector_ops.hpp"

#define sgn(a) (copysign(1.f,a))

using namespace cuv;

template<class T>
struct uf_exp{  __host__ __device__ T operator()(const T& t)const{ return __expf(t);    } };
template<class T>
struct uf_exact_exp{  __device__ __host__ T operator()(const T& t)const{ return exp(t);    } };
template<class T>
struct uf_log{  __device__ __host__ T operator()(const T& t)      const{ return log(t);    } };
template<class T>
struct uf_sign{  __device__         T operator()(const T& t)      const{ return sgn(t);    } };
template<class T>
struct uf_sigm{  __device__         T operator()(const T& t)      const{ return ((T)1)/(((T)1)+__expf(-t));    } };
template<class T>
struct uf_exact_sigm{  __device__  __host__ T operator()(const T& t)      const{ return ((T)1)/(((T)1)+exp(-t));    } };
template<class T>
struct uf_dsigm{  __device__ __host__       T operator()(const T& t)      const{ return t * (((T)1)-t); } };
template<class T>
struct uf_tanh{  __device__  __host__       T operator()(const T& t)      const{ return tanh(t); } };
template<class T>
struct uf_dtanh{  __device__  __host__      T operator()(const T& t)      const{ return ((T)1) - (t*t); } };
template<class T>
struct uf_square{  __device__  __host__     T operator()(const T& t)      const{ return t*t; } };
template<class T>
struct uf_sublin{  __device__  __host__     T operator()(const T& t)      const{ return ((T)1)-t; } };
template<class T>
struct uf_energ{  __device__  __host__      T operator()(const T& t)      const{ return -log(t); } };
template<class T>
struct uf_inv{  __device__  __host__        T operator()(const T& t)      const{ return ((T)1)/(t+((T)0.00000001)); } };
template<class T>
struct uf_sqrt{  __device__  __host__       T operator()(const T& t)      const{ return sqrt(t); } };

template<class T, class binary_functor>
struct uf_base_op{
  const T x;
  const binary_functor bf;
  uf_base_op(const T& _x):x(_x),bf(){};
  T operator()(T t){ return bf(t,x); }
};


template<class unary_functor, class value_type, class index_type>
void launch_unary_kernel(
   cuv::dev_vector<value_type, index_type>& dst,
   cuv::dev_vector<value_type, index_type>& src, 
	 unary_functor uf){
	 cuvAssert(dst.ptr());
	 cuvAssert(src.ptr());
	 cuvAssert(dst.size() == src.size());

	 thrust::device_ptr<value_type> dst_ptr(dst.ptr());
	 thrust::device_ptr<value_type> src_ptr(src.ptr());
	 thrust::transform(src_ptr,src_ptr+src.size(),dst_ptr,uf);
	 cuvSafeCall(cudaThreadSynchronize());
}

template<class unary_functor, class value_type, class index_type>
void launch_unary_kernel(
   cuv::host_vector<value_type, index_type>& dst,
   cuv::host_vector<value_type, index_type>& src, 
	 unary_functor uf){
	 cuvAssert(src.ptr());
	 cuvAssert(dst.ptr());
	 cuvAssert(dst.size() == src.size());
	 for(size_t i=0;i<dst.size();i++)
	   dst[i] = uf(src[i]);
}

namespace cuv{

template<class __vector_type>
struct apply_scalar_functor_impl;

template<class __vector_type>
void
apply_scalar_functor(__vector_type& v, const ScalarFunctor& sf){
  apply_scalar_functor_impl<__vector_type>::apply(v,sf);
}
template<class __vector_type, class __value_type>
void
apply_scalar_functor(__vector_type& v, const ScalarFunctor& sf, const __value_type& param){
  apply_scalar_functor_impl<__vector_type>::apply(v,sf,param);
}

template<class __vector_type>
struct apply_scalar_functor_impl{

	template<class __arg_value_type>
	static void
	apply(__vector_type& v, const ScalarFunctor& sf, const __arg_value_type& param){
		typedef typename __vector_type::value_type value_type;
		switch(sf){
			case SF_ADD:       launch_unary_kernel(v,v,uf_base_op<value_type, thrust::plus<value_type> >(param)); break;
			case SF_MULT:      launch_unary_kernel(v,v,uf_base_op<value_type, thrust::multiplies<value_type> >(param)); break;
			case SF_DIV:       launch_unary_kernel(v,v,uf_base_op<value_type, thrust::divides<value_type> >(param)); break;
			case SF_SUBTRACT:  launch_unary_kernel(v,v,uf_base_op<value_type, thrust::minus<value_type> >(param)); break;
		}
	}

	static void
	apply(__vector_type& v, const ScalarFunctor& sf){
		typedef typename __vector_type::value_type value_type;
	  switch(sf){
			case SF_EXP:        launch_unary_kernel(v,v, uf_exp<value_type>()); break;
			case SF_EXACT_EXP:  launch_unary_kernel(v,v, uf_exact_exp<value_type>()); break;
			case SF_LOG:        launch_unary_kernel(v,v, uf_log<value_type>()); break;
			case SF_SIGN:       launch_unary_kernel(v,v, uf_sign<value_type>()); break;
			case SF_SIGM:       launch_unary_kernel(v,v, uf_sigm<value_type>()); break;
			case SF_DSIGM:      launch_unary_kernel(v,v, uf_dsigm<value_type>()); break;
			case SF_TANH:       launch_unary_kernel(v,v, uf_tanh<value_type>()); break;
			case SF_DTANH:      launch_unary_kernel(v,v, uf_dtanh<value_type>()); break;
			case SF_SQUARE:     launch_unary_kernel(v,v, uf_square<value_type>()); break;
			case SF_SUBLIN:     launch_unary_kernel(v,v, uf_sublin<value_type>()); break;
			case SF_ENERG:      launch_unary_kernel(v,v, uf_energ<value_type>()); break;
			case SF_INV:        launch_unary_kernel(v,v, uf_inv<value_type>()); break;
			case SF_SQRT:       launch_unary_kernel(v,v, uf_sqrt<value_type>()); break;
			case SF_NEGATE:     launch_unary_kernel(v,v, thrust::negate<value_type>()); break;
			default:
			 cuvAssert(false);
		}
	}
};

template void apply_scalar_functor<dev_vector<float> >(dev_vector<float>&, const ScalarFunctor&);
template void apply_scalar_functor<dev_vector<float>, float>(dev_vector<float>&, const ScalarFunctor&,const float&);
template void apply_scalar_functor<dev_vector<float>, int>(dev_vector<float>&, const ScalarFunctor&,const int&);
} // cuv
