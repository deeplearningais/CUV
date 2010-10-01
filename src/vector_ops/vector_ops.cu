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





#include <cmath>
#include <iostream>
#include <cublas.h>

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
#include <thrust/generate.h>
#include <thrust/logical.h>

#include <cuv_general.hpp>
#include <cutil_inline.h>

#include <vector.hpp>

#include "vector_ops.hpp"

#define sgn(a) (copysign(1.f,a))

/*
 * USE_THRUST_LAUNCHER:
 * thrust has an overhead for looking up the correct block/grid-size for threads.
 * this overhead goes away for matrices of about 784*2048 for very simple linear kernels,
 * then they are better on bigcuda1.
 *
 */
#define USE_THRUST_LAUNCHER 0 


using namespace cuv;
using namespace std;

template<class T, class M>
struct memspace_cuv2thrustptr                          { typedef thrust::device_ptr<T> ptr_type; };
template<class T>
struct memspace_cuv2thrustptr<T,cuv::host_memory_space>{ typedef T* ptr_type; };
template<class T>
struct memspace_cuv2thrustptr<T,cuv::dev_memory_space> { typedef thrust::device_ptr<T> ptr_type; };

template<class T>
struct uf_exp{  __host__ __device__         T operator()(const T& t)const{ return expf(t);    } };
template<class T>
struct uf_exact_exp{  __device__ __host__   T operator()(const T& t)const{ return expf(t);    } };
template<class T>
struct uf_log{  __device__ __host__         T operator()(const T& t)      const{ return logf(t);    } };
template<class T>
struct uf_sign{  __device__ __host__        T operator()(const T& t)      const{ return sgn((float)t);    } };
template<class T>

#ifdef __DEVICE_EMULATION__
	struct uf_sigm{  __device__  __host__       T operator()(const T& t)      const{ return ((T)1)/(((T)1)+expf(-t));    } };
#else
	struct uf_sigm{  __device__  __host__       T operator()(const T& t)      const{ return ((T)1)/(((T)1)+expf(-t));    } };
#endif

template<class T>
struct uf_exact_sigm{  __device__  __host__ T operator()(const T& t)      const{ return ((T)1)/(((T)1)+expf(-t));    } };
template<class T>
struct uf_dsigm{  __device__ __host__       T operator()(const T& t)      const{ return t * (((T)1)-t); } };
template<class T>
struct uf_tanh{  __device__  __host__       T operator()(const T& t)      const{ return tanhf(t); } };
template<class T>
struct uf_dtanh{  __device__  __host__      T operator()(const T& t)      const{ return ((T)1) - (t*t); } };
template<class T>
struct uf_square{  __device__  __host__     T operator()(const T& t)      const{ return t*t; } };
template<class T>
struct uf_sublin{  __device__  __host__     T operator()(const T& t)      const{ return ((T)1)-t; } };
template<class T>
struct uf_energ{  __device__  __host__      T operator()(const T& t)      const{ return -logf(t); } };
template<class T>
struct uf_inv{  __device__  __host__        T operator()(const T& t)      const{ return ((T)1)/(t+((T)0.00000001)); } };
template<class T>
struct uf_sqrt{  __device__  __host__       T operator()(const T& t)      const{ return sqrtf(t); } };
template<class T>
struct uf_abs{  __device__  __host__       T operator()(const T& t)      const{ return t < T(0) ? -t : t; } };
template<class T>
struct uf_smax{  __device__  __host__      T operator()(const T& t)      const{ return (((T)1)/t - (T) 1) * t; } };

template<class T>
struct uf_is_nan{                 __device__  __host__   bool operator()(const T& t)             const{ return (t!=t) ; } };
template<>
struct uf_is_nan<int>{            __device__  __host__   bool operator()(const int& t)           const{ return false ; } };
template<>
struct uf_is_nan<unsigned char>{  __device__  __host__   bool operator()(const unsigned char& t) const{ return false ; } };
template<>
struct uf_is_nan<signed char>{    __device__  __host__   bool operator()(const signed char& t)   const{ return false ; } };

template<class T>
struct uf_is_inf{                 __device__  __host__     bool operator()(const T& t)            const{ return (bool)!isfinite(t); } };
template<>                                                                                        
struct uf_is_inf<int>{            __device__  __host__     bool operator()(const int t)           const{ return false; } };
template<>                                                                                        
struct uf_is_inf<signed char>{    __device__  __host__     bool operator()(const signed char t)   const{ return false; } };
template<>
struct uf_is_inf<unsigned char>{  __device__  __host__     bool operator()(const unsigned char t) const{ return false; } };

template<class T>
struct uf_poslin{  __device__  __host__     T operator()(const T& t)      const{ return (t > 0)*t; } };


template<class T>
struct bf_sigm_temp{ __device__  __host__       T operator()(const T& t, const T& temp)           const{ return ((T)1)/(((T)1)+expf(-t / (T)(temp))); } };

template<class T>
struct tf_tanh{  __device__  __host__       T operator()(const T& x, const T& a, const T& b)      const{ return a * tanhf(b * x); } };
template<class T>
struct tf_dtanh{  __device__  __host__      T operator()(const T& x, const T& a, const T& b)      const{ return b/a * (a+x) * (a-x); } };

// rectifying transferfunctions a is param beta
template<class T, class A>
struct tf_rect{  __device__  __host__       T operator()(const T& x, const A& a)      const{
	T ax = a*x;
	if(-ax < -87.33f)
		return (T) 0;
	return log(1.0f+expf(ax))/a;
}};
/*template<class T, class A>*/
/*struct tf_rect{  __device__  __host__       T operator()(const T& x, const A& a)      const{ return (T) log(1.0 + (double)exp((double)a*x))/a; } };*/
template<class T, class A>
struct tf_drect{  __device__  __host__      T operator()(const T& x, const A& a)      const{ return 1-1/(x*expf(a)); } };


template<class T, class binary_functor>
struct uf_base_op{
  const T x;
  const binary_functor bf;
  uf_base_op(const T& _x):x(_x),bf(){};
  __device__ __host__
  T operator()(const T& t){ return bf(t,x); }
};
template<class T, class ternary_functor>
struct uf_base_op3{
  const T x,y;
  const ternary_functor tf;
  uf_base_op3(const T& _x, const T& _y):x(_x),y(_y),tf(){};
  __device__ __host__
  T operator()(const T& t){ return tf(t,x,y); }
};

/*
 * Binary Functors
 */

// functors without parameter
template<class T, class U>
struct bf_plus{  __device__  __host__       T operator()(const T& t, const U& u)      const{ return  t + (T)u; } };
template<class T, class U>
struct bf_minus{  __device__  __host__      T operator()(const T& t, const U& u)      const{ return  t - (T)u; } };
template<class T, class U>
struct bf_multiplies{  __device__  __host__ T operator()(const T& t, const U& u)      const{ return  t * (T)u; } };
template<class T, class U>
struct bf_divides{  __device__  __host__    T operator()(const T& t, const U& u)      const{ return  t / (T)u; } };
template<class T, class U>
struct bf_squared_diff{__device__ __host__  T operator()(const T& t, const U& u)      const{ T ret =  t - (T)u; return ret*ret; } };
template<class T, class U>
struct bf_and{__device__ __host__   T operator()(const T& t, const U& u)      const{ return t && u; } };
template<class T, class U>
struct bf_or{ __device__ __host__   T operator()(const T& t, const U& u)      const{ return t || u; } };
template<class T, class U>
struct bf_min{ __device__ __host__  T operator()(const T& t, const U& u)      const{ return t<u ? t : u; } };
template<class T, class U>
struct bf_max{ __device__ __host__  T operator()(const T& t, const U& u)      const{ return t>u ? t : u; } };

// functors with parameter
template<class T, class U>
struct bf_axpy{  
	const T a;
	bf_axpy(const T& _a):a(_a){}
	__device__  __host__       T operator()(const T& t, const U& u) const{ return  a*t+(T)u; } 
};
template<class T, class U>
struct bf_xpby{  
	const T b;
	bf_xpby(const T& _b):b(_b){}
	__device__  __host__       T operator()(const T& t, const U& u) const{ return  t+b*(T)u; } 
};
template<class T, class U>
struct bf_axpby{  
	const T a;
	const T b;
	bf_axpby(const T& _a, const T& _b):a(_a),b(_b){}
	__device__  __host__       T operator()(const T& t, const U& u) const{ return  a*t + b*((T)u); } 
};


/*
 * launchers for functors
 */

#if ! USE_THRUST_LAUNCHER
template<class unary_functor, class value_type, class index_type>
__global__
void unary_functor_kernel(value_type* dst, value_type* src, index_type n, unary_functor uf){
	const unsigned int idx = __mul24(blockIdx.x , blockDim.x) + threadIdx.x;
	const unsigned int off = __mul24(blockDim.x , gridDim.x);
	for (unsigned int i = idx; i < n; i += off)
		dst[i] = uf(src[i]);
}

template<class binary_functor, class value_type, class value_type2, class index_type>
__global__
void binary_functor_kernel(value_type* dst, value_type* src, value_type2* src2, index_type n, binary_functor bf){
	const unsigned int idx = __mul24(blockIdx.x , blockDim.x) + threadIdx.x;
	const unsigned int off = __mul24(blockDim.x , gridDim.x);
	for (unsigned int i = idx; i < n; i += off)
		dst[i] = bf(src[i],src2[i]);
}

void setLinearGridAndThreads(dim3& blocks, dim3& threads, size_t len, int threads_per_block=512){
	const int padded_len=(int)ceil((float)len/threads_per_block)*threads_per_block;
	blocks = dim3(min(512,padded_len/threads_per_block),1,1);
	threads = dim3(threads_per_block,1,1);
}
#endif

template<class unary_functor, class value_type, class index_type>
void launch_unary_kernel(
   cuv::vector<value_type,dev_memory_space,index_type>& dst,
   cuv::vector<value_type,dev_memory_space,index_type>& src, 
	 unary_functor uf){
	 cuvAssert(dst.ptr());
	 cuvAssert(src.ptr());
	 cuvAssert(dst.size() == src.size());

#if ! USE_THRUST_LAUNCHER
	 dim3 blocks, threads;
	 setLinearGridAndThreads(blocks,threads,dst.size());
	 unary_functor_kernel<<<blocks,threads>>>(dst.ptr(),src.ptr(),dst.size(),uf); //     180 ms
#else
	 thrust::device_ptr<value_type> dst_ptr(dst.ptr());
	 thrust::device_ptr<value_type> src_ptr(src.ptr());
	 thrust::transform(src_ptr,src_ptr+src.size(),dst_ptr,uf);
#endif

	 cuvSafeCall(cudaThreadSynchronize());
}

template<class unary_functor, class value_type, class index_type>
void launch_unary_kernel(
   cuv::vector<value_type,host_memory_space,index_type>& dst,
   cuv::vector<value_type,host_memory_space,index_type>& src, 
	 unary_functor uf){
	 cuvAssert(src.ptr());
	 cuvAssert(dst.ptr());
	 cuvAssert(dst.size() == src.size());
	 value_type* dst_ptr = dst.ptr();
	 value_type* src_ptr = src.ptr();
	 for(size_t i=0;i<dst.size();i++)
	   *dst_ptr++ = uf( *src_ptr++ );
}

template<class binary_functor, class V1, class V2, class index_type>
void launch_binary_kernel(
   cuv::vector<V1,dev_memory_space,index_type>& v,
   cuv::vector<V2,dev_memory_space,index_type>& w, 
	 binary_functor bf){
	 cuvAssert(v.ptr());
	 cuvAssert(w.ptr());
	 cuvAssert(v.size() == w.size());

#if ! USE_THRUST_LAUNCHER
	 dim3 blocks, threads;
	 setLinearGridAndThreads(blocks,threads,v.size());
	 binary_functor_kernel<<<blocks,threads>>>(v.ptr(),v.ptr(),w.ptr(),v.size(),bf); 
#else
	 thrust::device_ptr<V1> v_ptr(v.ptr());
	 thrust::device_ptr<V2> w_ptr(w.ptr());
	 thrust::transform(v_ptr,v_ptr+v.size(),w_ptr,bf);
#endif

	 cuvSafeCall(cudaThreadSynchronize());
}

template<class binary_functor, class V1, class V2, class index_type>
void launch_binary_kernel(
   cuv::vector<V1,host_memory_space,index_type>& dst,
   cuv::vector<V2,host_memory_space,index_type>& src, 
	 binary_functor uf){
	 cuvAssert(src.ptr());
	 cuvAssert(dst.ptr());
	 cuvAssert(dst.size() == src.size());
	 V1* dst_ptr = dst.ptr();
	 V2* src_ptr = src.ptr();
	 for(size_t i=0;i<dst.size();i++)
	   *dst_ptr++ = uf(*dst_ptr,*src_ptr++);
}

namespace cuv{
	
/*
 * Nullary Functor
 *
 */

template<class __vector_type>
void
apply_0ary_functor(__vector_type& v, const NullaryFunctor& nf){
	 cuvAssert(v.ptr());
	 typedef typename __vector_type::value_type value_type;
	 typedef typename memspace_cuv2thrustptr<value_type,typename __vector_type::memory_space_type>::ptr_type ptr_type;
	 ptr_type dst_ptr(v.ptr());
	 switch(nf){
		 case NF_SEQ:
			 thrust::sequence(dst_ptr,dst_ptr+v.size());break;
		 default:
			 cuvAssert(false);
	 }
	 cuvSafeCall(cudaThreadSynchronize());
}

template<class __vector_type, class __value_type>
void
apply_0ary_functor(__vector_type& v, const NullaryFunctor& nf, const __value_type& param){
	 cuvAssert(v.ptr());

	 typedef typename __vector_type::value_type value_type;
	 typedef typename memspace_cuv2thrustptr<value_type,typename __vector_type::memory_space_type>::ptr_type ptr_type;
	 ptr_type dst_ptr(v.ptr());
	 switch(nf){
		 case NF_FILL:
			 thrust::fill(dst_ptr,dst_ptr + v.size(), (value_type)param); break;
		 default:
			 cuvAssert(false);
	 }
	 cuvSafeCall(cudaThreadSynchronize());
}

/*
 * Unary Functor
 *
 */
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
template<class __vector_type, class __value_type>
void
apply_scalar_functor(__vector_type& v, const ScalarFunctor& sf, const __value_type& param, const __value_type& param2){
  apply_scalar_functor_impl<__vector_type>::apply(v,sf,param,param2);
}

/*
 * Binary Functor
 *
 */
template<class __vector_type1, class __vector_type2>
void
apply_binary_functor(__vector_type1& v, __vector_type2& w, const BinaryFunctor& sf){
	cuvAssert(v.size() == w.size());
	typedef typename __vector_type1::value_type V1;
	typedef typename __vector_type2::value_type V2;
	typedef typename memspace_cuv2thrustptr<V1,typename __vector_type1::memory_space_type>::ptr_type ptr_type1;
	typedef typename memspace_cuv2thrustptr<V2,typename __vector_type2::memory_space_type>::ptr_type ptr_type2;
	ptr_type1 v_ptr(v.ptr());
	ptr_type2 w_ptr(w.ptr());
#if USE_THRUST_LAUNCHER 
	switch(sf){
		case BF_ADD:      thrust::transform(v_ptr, v_ptr+v.size(), w_ptr,  v_ptr, bf_plus<V1,V2>()); break;
		case BF_SUBTRACT: thrust::transform(v_ptr, v_ptr+v.size(), w_ptr,  v_ptr, bf_minus<V1,V2>()); break;
		case BF_MULT:     thrust::transform(v_ptr, v_ptr+v.size(), w_ptr,  v_ptr, bf_multiplies<V1,V2>()); break;
		case BF_DIV:      thrust::transform(v_ptr, v_ptr+v.size(), w_ptr,  v_ptr, bf_divides<V1,V2>()); break;
		case BF_MIN:      thrust::transform(v_ptr, v_ptr+v.size(), w_ptr,  v_ptr, bf_min<V1,V2>()); break;
		case BF_MAX:      thrust::transform(v_ptr, v_ptr+v.size(), w_ptr,  v_ptr, bf_max<V1,V2>()); break;
		case BF_COPY:     thrust::copy(w_ptr, w_ptr+v.size(), v_ptr); break;
		default: cuvAssert(false);
	}
#else
	dim3 blocks, threads;
	setLinearGridAndThreads(blocks,threads,v.size());
	switch(sf){
		case BF_ADD:      launch_binary_kernel(v,w,bf_plus<V1,V2>()); break;
		case BF_SUBTRACT: launch_binary_kernel(v,w,bf_minus<V1,V2>()); break;
		case BF_MULT:     launch_binary_kernel(v,w,bf_multiplies<V1,V2>()); break;
		case BF_DIV:      launch_binary_kernel(v,w,bf_divides<V1,V2>()); break;
		case BF_MIN:      launch_binary_kernel(v,w,bf_min<V1,V2>()); break;
		case BF_MAX:      launch_binary_kernel(v,w,bf_max<V1,V2>()); break;
		case BF_COPY:     thrust::copy(w_ptr, w_ptr+v.size(), v_ptr); break;
		default: cuvAssert(false);
	}
#endif
	cuvSafeCall(cudaThreadSynchronize());
}

template<class __vector_type1, class __vector_type2, class __value_type>
void
apply_binary_functor(__vector_type1& v, __vector_type2& w, const BinaryFunctor& sf, const __value_type& param){
	cuvAssert(v.size() == w.size());
	typedef typename __vector_type1::value_type V1;
	typedef typename __vector_type2::value_type V2;
	typedef typename memspace_cuv2thrustptr<V1,typename __vector_type1::memory_space_type>::ptr_type ptr_type1;
	typedef typename memspace_cuv2thrustptr<V2,typename __vector_type2::memory_space_type>::ptr_type ptr_type2;
	ptr_type1 v_ptr(v.ptr());
	ptr_type2 w_ptr(w.ptr());
#if USE_THRUST_LAUNCHER
	switch(sf){
		case BF_AXPY:     thrust::transform(v_ptr, v_ptr+v.size(), w_ptr,  v_ptr, bf_axpy<V1,V2>(param)); break;
		case BF_XPBY:     thrust::transform(v_ptr, v_ptr+v.size(), w_ptr,  v_ptr, bf_xpby<V1,V2>(param)); break;
		/*case BF_XPBY:     cublasSaxpy(v.size(), param, (float*)w.ptr(), 1, (float*)v.ptr(), 1) ; break;*/
		default: cuvAssert(false);
	}
#else
	dim3 blocks, threads;
	setLinearGridAndThreads(blocks,threads,v.size());
	switch(sf){
		case BF_AXPY:     launch_binary_kernel(v,w,bf_axpy<V1,V2>(param)); break;
		case BF_XPBY:     launch_binary_kernel(v,w,bf_xpby<V1,V2>(param)); break;
		default: cuvAssert(false);
	}
#endif
	cuvSafeCall(cudaThreadSynchronize());
}

template<class __vector_type1, class __vector_type2, class __value_type>
void
apply_binary_functor(__vector_type1& v, __vector_type2& w, const BinaryFunctor& sf, const __value_type& param, const __value_type& param2){
	cuvAssert(v.size() == w.size());
	typedef typename __vector_type1::value_type V1;
	typedef typename __vector_type2::value_type V2;
	typedef typename memspace_cuv2thrustptr<V1,typename __vector_type1::memory_space_type>::ptr_type ptr_type1;
	typedef typename memspace_cuv2thrustptr<V2,typename __vector_type2::memory_space_type>::ptr_type ptr_type2;
	ptr_type1 v_ptr(v.ptr());
	ptr_type2 w_ptr(w.ptr());
#if USE_THRUST_LAUNCHER
	switch(sf){
		case BF_AXPBY:     thrust::transform(v_ptr, v_ptr+v.size(), w_ptr,  v_ptr, bf_axpby<V1,V2>(param,param2)); break;
		default: cuvAssert(false);
	}
#else
	dim3 blocks, threads;
	setLinearGridAndThreads(blocks,threads,v.size());
	switch(sf){
		case BF_AXPBY:     launch_binary_kernel(v,w,bf_axpby<V1,V2>(param,param2)); break;
		default: cuvAssert(false);
	}
#endif
	cuvSafeCall(cudaThreadSynchronize());
}

template<class __vector_type>
struct apply_scalar_functor_impl{


	template<class __arg_value_type>
	static void
	apply(__vector_type& v, const ScalarFunctor& sf, const __arg_value_type& param, const __arg_value_type& param2){
		typedef typename __vector_type::value_type value_type;
		switch(sf){
			case SF_TANH:      launch_unary_kernel(v,v,uf_base_op3<value_type, tf_tanh<value_type> >(param,param2)); break;
			case SF_DTANH:     launch_unary_kernel(v,v,uf_base_op3<value_type, tf_dtanh<value_type> >(param,param2)); break;
			default:
				cuvAssert(false);
		}
	}

	template<class __arg_value_type>
	static void
	apply(__vector_type& v, const ScalarFunctor& sf, const __arg_value_type& param){
		typedef typename __vector_type::value_type value_type;
		switch(sf){
			case SF_SIGM:      launch_unary_kernel(v,v,uf_base_op<value_type, bf_sigm_temp<value_type> >(param)); break;
			case SF_ADD:       launch_unary_kernel(v,v,uf_base_op<value_type, thrust::plus<value_type> >(param)); break;
			case SF_MULT:      launch_unary_kernel(v,v,uf_base_op<value_type, thrust::multiplies<value_type> >(param)); break;
			case SF_DIV:       launch_unary_kernel(v,v,uf_base_op<value_type, thrust::divides<value_type> >(param)); break;
			case SF_SUBTRACT:  launch_unary_kernel(v,v,uf_base_op<value_type, thrust::minus<value_type> >(param)); break;
			case SF_MIN:       launch_unary_kernel(v,v,uf_base_op<value_type, bf_min<value_type,__arg_value_type> >(param)); break;
			case SF_MAX:       launch_unary_kernel(v,v,uf_base_op<value_type, bf_max<value_type,__arg_value_type> >(param)); break;
			case SF_RECT:      launch_unary_kernel(v,v,uf_base_op<value_type, tf_rect<value_type,__arg_value_type> >(param)); break;
			case SF_DRECT:     launch_unary_kernel(v,v,uf_base_op<value_type, tf_drect<value_type,__arg_value_type> >(param)); break;
			default:
				cuvAssert(false);
		}
	}

	static void
	apply(__vector_type& v, const ScalarFunctor& sf){
		typedef typename __vector_type::value_type value_type;
	  switch(sf){
			case SF_EXP:        launch_unary_kernel(v,v, uf_exp<value_type>()); break;
			/*case SF_EXACT_EXP:  launch_unary_kernel(v,v, uf_exact_exp<value_type>()); break;*/
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
			case SF_SMAX:       launch_unary_kernel(v,v, uf_smax<value_type>()); break;
			case SF_NEGATE:     launch_unary_kernel(v,v, thrust::negate<value_type>()); break;
			case SF_POSLIN:     launch_unary_kernel(v,v, uf_poslin<value_type>()); break;
			default:
			 cuvAssert(false);
		}
	}
};

/*
 * Reductions
 */
template<class __vector_type>
bool
has_inf(__vector_type& v){
	typedef typename __vector_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __vector_type::memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(v.ptr());
	bool init=false;
	uf_is_inf<value_type> uo;
	return  thrust::any_of(v_ptr, v_ptr+v.size(), uo);
}
template<class __vector_type>
bool
has_nan(__vector_type& v){
	typedef typename __vector_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __vector_type::memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(v.ptr());
	bool init=false;
	uf_is_nan<value_type> uo;
	return  thrust::any_of(v_ptr, v_ptr+v.size(), uo);
}
template<class __vector_type>
float
norm2(__vector_type& v){
	typedef typename __vector_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __vector_type::memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(v.ptr());
	float init=0;
	return  std::sqrt( thrust::transform_reduce(v_ptr, v_ptr+v.size(), uf_square<float>(), init, bf_plus<float,value_type>()) );
}
template<class __vector_type>
float
norm1(__vector_type& v){
	typedef typename __vector_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __vector_type::memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(v.ptr());
	float init=0;
	uf_abs<float> unary_op;
	bf_plus<float,value_type> binary_op;
	return   thrust::transform_reduce(v_ptr, v_ptr+v.size(), unary_op, init, binary_op);
}
template<class __vector_type>
float
sum(__vector_type& v){
	typedef typename __vector_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __vector_type::memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(v.ptr());
	float init=0.0;
	return   thrust::reduce(v_ptr, v_ptr+v.size(), init, bf_plus<float,value_type>());
}
template<class __vector_type>
float
maximum(__vector_type& v){
	typedef typename __vector_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __vector_type::memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(v.ptr());
	float init=-INT_MAX;
	return   thrust::reduce(v_ptr, v_ptr+v.size(), init, bf_max<float,value_type>());
}
template<class __vector_type>
float
minimum(__vector_type& v){
	typedef typename __vector_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __vector_type::memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(v.ptr());
	float init=INT_MAX;
	return   thrust::reduce(v_ptr, v_ptr+v.size(), init, bf_min<float,value_type>());
}
template<class __vector_type>
float
mean(__vector_type& v){
	return   sum(v) / (float)v.size();
}
template<class __vector_type>
float
var(__vector_type& v){
	typedef typename __vector_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __vector_type::memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(v.ptr());
	float init=0;
	float m = mean(v);
	return   thrust::transform_reduce(v_ptr, v_ptr+v.size(), uf_base_op<float, bf_squared_diff<float,value_type> >(m), init, bf_plus<float,value_type>()) / (float)v.size();
}
template<class __vector_type>
typename __vector_type::index_type
arg_max(__vector_type& v){
	typedef typename __vector_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __vector_type::memory_space_type>::ptr_type ptr_type;
	ptr_type begin(v.ptr());
	ptr_type elem = thrust::max_element(begin, begin	+v.size());
	return thrust::distance(begin,elem);
}
template<class __vector_type>
typename __vector_type::index_type
arg_min(__vector_type& v){
	typedef typename __vector_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __vector_type::memory_space_type>::ptr_type ptr_type;
	ptr_type begin(v.ptr());
	ptr_type elem = thrust::min_element(begin, begin	+v.size());
	return thrust::distance(begin,elem);
}
/*
 * Template instantiations
 */

#define SIMPLE_0(X,Y) \
	template void apply_0ary_functor< vector<X,Y> >( vector<X,Y>&, const NullaryFunctor&);

#define SIMPLE_01(X,P,Z) \
	template void apply_0ary_functor< vector<X,Z>, P>(vector<X,Z>&, const NullaryFunctor&, const P& param);

#define SIMPLE_1(X,Y) \
	template void apply_scalar_functor< vector<X,Y> >(vector<X,Y>&, const ScalarFunctor&);
#define SIMPLE_11(X,P,Z) \
	template void apply_scalar_functor< vector<X,Z>, P>(vector<X,Z>&, const ScalarFunctor&,const P&); \
	template void apply_scalar_functor< vector<X,Z>, P>(vector<X,Z>&, const ScalarFunctor&,const P&, const P&);

#define SIMPLE_2(X,Y,Z) \
	template void apply_binary_functor< vector<X,Z> , vector<Y,Z>  >( vector<X,Z> &,  vector<Y,Z> &, const BinaryFunctor&);
#define SIMPLE_21(X,Y,P,Z) \
	template void apply_binary_functor<vector<X,Z>,vector<Y,Z>,P>(vector<X,Z>&, vector<Y,Z>&, const BinaryFunctor&,  const P&); \
	template void apply_binary_functor<vector<X,Z>,vector<Y,Z>,P>(vector<X,Z>&, vector<Y,Z>&, const BinaryFunctor&,  const P&, const P&);

#define SIMPLE_NORM(X, Y) \
	template bool has_inf<vector<X,Y> >(vector<X,Y>&); \
	template bool has_nan<vector<X,Y> >(vector<X,Y>&); \
	template float minimum<vector<X,Y> >(vector<X,Y>&); \
	template float maximum<vector<X,Y> >(vector<X,Y>&); \
	template float norm1<vector<X,Y> >(vector<X,Y>&); \
	template float norm2<vector<X,Y> >(vector<X,Y>&); \
	template float mean<vector<X,Y> >(vector<X,Y>&);  \
	template float var<vector<X,Y> >(vector<X,Y>&); \
	template typename vector<X,Y>::index_type     arg_max<vector<X,Y> >(vector<X,Y>&); \
	template typename vector<X,Y>::index_type     arg_min<vector<X,Y> >(vector<X,Y>&);


#define SIMPLE_INSTANTIATOR(X) \
	SIMPLE_0( X , dev_memory_space);             \
	SIMPLE_1( X , dev_memory_space);             \
	SIMPLE_2( X, X , dev_memory_space);          \
    SIMPLE_NORM( X , dev_memory_space);			\
	SIMPLE_0( X , host_memory_space);             \
	SIMPLE_1( X , host_memory_space);             \
	SIMPLE_2( X, X , host_memory_space);          \
    SIMPLE_NORM( X , host_memory_space);

#define SIMPLE_INSTANTIATOR1(X, P) \
	SIMPLE_01( X, P , dev_memory_space);             \
	SIMPLE_11( X, P , dev_memory_space);             \
	SIMPLE_21( X, X, P , dev_memory_space);          \
	SIMPLE_01( X, P , host_memory_space);             \
	SIMPLE_11( X, P , host_memory_space);             \
	SIMPLE_21( X, X, P , host_memory_space);          

SIMPLE_INSTANTIATOR( float );
SIMPLE_INSTANTIATOR1( float, float );
SIMPLE_INSTANTIATOR1( float, int );
SIMPLE_INSTANTIATOR( unsigned char );
SIMPLE_INSTANTIATOR1( unsigned char, unsigned char );
SIMPLE_INSTANTIATOR( signed char );
SIMPLE_INSTANTIATOR1( signed char, unsigned char );
SIMPLE_2(float,unsigned char, dev_memory_space);
SIMPLE_2(float,unsigned char, host_memory_space);
SIMPLE_INSTANTIATOR(int );
SIMPLE_INSTANTIATOR1(int, int );
} // cuv
