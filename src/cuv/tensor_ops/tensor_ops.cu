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

#include <cuv/tools/cuv_general.hpp>

#include <cuv/basics/tensor.hpp>
#include <cuv/tensor_ops/functors.hpp>

#include <cuv/tensor_ops/tensor_ops.hpp>


/*
 * USE_THRUST_LAUNCHER:
 * thrust has an overhead for looking up the correct block/grid-size for threads.
 * this overhead goes away for matrices of about 784*2048 for very simple linear kernels,
 * then they are better on bigcuda1.
 *
 */
#define USE_THRUST_LAUNCHER 1 


using namespace cuv;
using namespace std;

template<class T, class M>
struct memspace_cuv2thrustptr                          { typedef thrust::device_ptr<T> ptr_type; };
template<class T>
struct memspace_cuv2thrustptr<T,cuv::host_memory_space>{ typedef T* ptr_type; };
template<class T>
struct memspace_cuv2thrustptr<T,cuv::dev_memory_space> { typedef thrust::device_ptr<T> ptr_type; };

/*
 * launchers for functors
 */

#if ! USE_THRUST_LAUNCHER
template<class unary_functor, class value_type, class index_type>
__global__
void unary_functor_kernel(value_type* dst, value_type* src n, unary_functor uf){
	const unsigned int idx = __mul24(blockIdx.x , blockDim.x) + threadIdx.x;
	const unsigned int off = __mul24(blockDim.x , gridDim.x);
	for (unsigned int i = idx; i < n; i += off)
		dst[i] = uf(src[i]);
}

template<class binary_functor, class value_type, class value_type2, class index_type>
__global__
void binary_functor_kernel(value_type* dst, value_type* src, value_type2* src2 n, binary_functor bf){
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

template<class unary_functor, class V1, class V2>
void launch_unary_kernel(
   cuv::tensor<V1,dev_memory_space>& dst,
   const cuv::tensor<V2,dev_memory_space>& src, 
	 unary_functor uf){
	 cuvAssert(dst.ptr());
	 cuvAssert(src.ptr());
	 cuvAssert(dst.size() == src.size());

#if ! USE_THRUST_LAUNCHER
	 dim3 blocks, threads;
	 setLinearGridAndThreads(blocks,threads,dst.size());
	 unary_functor_kernel<<<blocks,threads>>>(dst.ptr(),src.ptr(),dst.size(),uf); //     180 ms
#else
	 thrust::device_ptr<V1> dst_ptr(dst.ptr());
	 thrust::device_ptr<V2> src_ptr(const_cast<V2*>(src.ptr()));
	 thrust::transform(src_ptr,src_ptr+src.size(),dst_ptr,uf);
#endif

	 cuvSafeCall(cudaThreadSynchronize());
}

template<class unary_functor, class V1, class V2>
void launch_unary_kernel(
   cuv::tensor<V1,host_memory_space>& dst,
   const cuv::tensor<V2,host_memory_space>& src, 
	 unary_functor uf){
	 cuvAssert(src.ptr());
	 cuvAssert(dst.ptr());
	 cuvAssert(dst.size() == src.size());
	 V1* dst_ptr = dst.ptr();
	 const V2* src_ptr = src.ptr();
	 for(size_t i=0;i<dst.size();i++)
	   *dst_ptr++ = uf( *src_ptr++ );
}

template<class binary_functor, class V1, class V2>
void launch_binary_kernel(
   cuv::tensor<V1,dev_memory_space>& v,
   const cuv::tensor<V2,dev_memory_space>& w, 
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

template<class binary_functor, class V1, class V2>
void launch_binary_kernel(
   cuv::tensor<V1,host_memory_space>& dst,
   const cuv::tensor<V2,host_memory_space>& src, 
	 binary_functor uf){
	 cuvAssert(src.ptr());
	 cuvAssert(dst.ptr());
	 cuvAssert(dst.size() == src.size());
	 V1* dst_ptr = dst.ptr();
	 const V2* src_ptr = src.ptr();
	 for(size_t i=0;i<dst.size();i++)
	   *dst_ptr++ = uf(*dst_ptr,*src_ptr++);
}

namespace cuv{
	
/*
 * Nullary Functor
 *
 */

template<class __tensor_type>
void
apply_0ary_functor(__tensor_type& v, const NullaryFunctor& nf){
	 cuvAssert(v.ptr());
	 typedef typename __tensor_type::value_type value_type;
	 typedef typename memspace_cuv2thrustptr<value_type,typename __tensor_type::memory_space_type>::ptr_type ptr_type;
	 ptr_type dst_ptr(v.ptr());
	 switch(nf){
		 case NF_SEQ:
			 thrust::sequence(dst_ptr,dst_ptr+v.size());break;
		 default:
			 cuvAssert(false);
	 }
	 cuvSafeCall(cudaThreadSynchronize());
}

template<class __tensor_type>
void
apply_0ary_functor(__tensor_type& v, const NullaryFunctor& nf, const typename __tensor_type::value_type& param){
	 cuvAssert(v.ptr());

	 typedef typename __tensor_type::value_type value_type;
	 typedef typename memspace_cuv2thrustptr<value_type,typename __tensor_type::memory_space_type>::ptr_type ptr_type;
	 ptr_type dst_ptr(v.ptr());
	 switch(nf){
		 case NF_FILL:
			 thrust::fill(dst_ptr,dst_ptr + v.size(), (value_type)param); break;
		 default:
			 cuvAssert(false);
	 }
	 cuvSafeCall(cudaThreadSynchronize());
}


namespace detail{
	// **********************************
	//       Scalar Functor
	// **********************************
	template<class D, class S, class V>
	void
	apply_scalar_functor(D&dst,const S&src, const ScalarFunctor& sf, const int& numparams, const V& p, const V& p2){
		cuvAssert(dst.size()==src.size());
		typedef typename S::value_type SV;
		typedef typename D::value_type DV;

		typedef typename memspace_cuv2thrustptr<DV, typename D::memory_space_type>::ptr_type ptr_type1;
		typedef typename memspace_cuv2thrustptr<SV, typename S::memory_space_type>::ptr_type ptr_type2;
		ptr_type1 d_ptr(dst.ptr());
		ptr_type2 s_ptr(const_cast<SV*>(src.ptr()));
		if(numparams==2){
			switch(sf){
				case SF_TANH:      launch_unary_kernel(dst,src,make_bind2nd3rd(tf_tanh <DV>(),p,p2)); break;
				case SF_DTANH:     launch_unary_kernel(dst,src,make_bind2nd3rd(tf_dtanh<DV>(),p,p2)); break;
				default:
						   cout << "No suitable two-parameter scalar functor was found." << endl;
						   cuvAssert(false);
			}
		}
		else if(numparams==1){
			switch(sf){
				case SF_SIGM:      launch_unary_kernel(dst,src,make_bind2nd(bf_sigm_temp<DV,SV>(),p)); break;
				case SF_ADD:       launch_unary_kernel(dst,src,make_bind2nd(thrust::plus<DV>(),p)); break;
				case SF_MULT:      launch_unary_kernel(dst,src,make_bind2nd(thrust::multiplies<DV>(),p)); break;
				case SF_DIV:       launch_unary_kernel(dst,src,make_bind2nd(thrust::divides<DV>(),p)); break;
				case SF_SUBTRACT:  launch_unary_kernel(dst,src,make_bind2nd(thrust::minus<DV>(),p)); break;
				case SF_MIN:       launch_unary_kernel(dst,src,make_bind2nd(bf_min<DV,SV,V>(),p)); break;
				case SF_MAX:       launch_unary_kernel(dst,src,make_bind2nd(bf_max<DV,SV,V>(),p)); break;
				case SF_RECT:      launch_unary_kernel(dst,src,make_bind2nd(bf_rect<DV,SV,V>(),p)); break;
				case SF_DRECT:     launch_unary_kernel(dst,src,make_bind2nd(bf_drect<DV,SV,V>(),p)); break;
				case SF_EQ:        launch_unary_kernel(dst,src,make_bind2nd(thrust::equal_to<SV>(),p)); break;
				case SF_LT:        launch_unary_kernel(dst,src,make_bind2nd(thrust::less<SV>(),p)); break;
				case SF_GT:        launch_unary_kernel(dst,src,make_bind2nd(thrust::greater<SV>(),p)); break;
				case SF_LEQ:       launch_unary_kernel(dst,src,make_bind2nd(thrust::less_equal<SV>(),p)); break;
				case SF_GEQ:       launch_unary_kernel(dst,src,make_bind2nd(thrust::greater_equal<SV>(),p)); break;
				default:
						   cout << "No suitable one-parameter scalar functor was found." << endl;
						   cuvAssert(false);
			}
		}
		else if(numparams==0){
			switch(sf){
				case SF_EXP:        launch_unary_kernel(dst,src, uf_exp<DV,SV>()); break;
				case SF_LOG:        launch_unary_kernel(dst,src, uf_log<DV,SV>()); break;
				case SF_SIGN:       launch_unary_kernel(dst,src, uf_sign<DV,SV>()); break;
				case SF_SIGM:       launch_unary_kernel(dst,src, uf_sigm<DV,SV>()); break;
				case SF_DSIGM:      launch_unary_kernel(dst,src, uf_dsigm<DV,SV>()); break;
				case SF_TANH:       launch_unary_kernel(dst,src, uf_tanh<DV,SV>()); break;
				case SF_DTANH:      launch_unary_kernel(dst,src, uf_dtanh<DV,SV>()); break;
				case SF_SQUARE:     launch_unary_kernel(dst,src, uf_square<DV,SV>()); break;
				case SF_SUBLIN:     launch_unary_kernel(dst,src, uf_sublin<DV,SV>()); break;
				case SF_ENERG:      launch_unary_kernel(dst,src, uf_energ<DV,SV>()); break;
				case SF_INV:        launch_unary_kernel(dst,src, uf_inv<DV,SV>()); break;
				case SF_SQRT:       launch_unary_kernel(dst,src, uf_sqrt<DV,SV>()); break;
				case SF_SMAX:       launch_unary_kernel(dst,src, uf_smax<DV,SV>()); break;
				case SF_NEGATE:     launch_unary_kernel(dst,src, thrust::negate<DV>()); break;
				case SF_ABS:        launch_unary_kernel(dst,src, uf_abs<DV,SV>()); break;
				case SF_POSLIN:     launch_unary_kernel(dst,src, uf_poslin<DV,SV>()); break;
				case SF_COPY:       thrust::copy(s_ptr, s_ptr+src.size(), d_ptr); break;
				default:
						    cout << "No suitable no-parameter scalar functor was found." << endl;
						    cuvAssert(false);
			}
		}
		cuvSafeCall(cudaThreadSynchronize());
	}


	// **********************************
	//       Binary Functor
	// **********************************
	template<class D, class S1, class S2, class V>
	void
	apply_binary_functor(D& dst,const S1&src1, const S2&src2, const BinaryFunctor& bf, const int& numparams, const V& p, const V& p2){
		cuvAssert(dst.size() == src1.size());
		cuvAssert(dst.size() == src2.size());
		typedef typename D::value_type  DV;
		typedef typename S1::value_type S1V;
		typedef typename S2::value_type S2V;
		typedef typename memspace_cuv2thrustptr<DV, typename  D::memory_space_type>::ptr_type ptr_type1;
		typedef typename memspace_cuv2thrustptr<S1V,typename S1::memory_space_type>::ptr_type ptr_type2;
		typedef typename memspace_cuv2thrustptr<S2V,typename S2::memory_space_type>::ptr_type ptr_type3;
		ptr_type1 d_ptr(dst.ptr());
		ptr_type2 s1_ptr(const_cast<S1V*>(src1.ptr()));
		ptr_type2 s2_ptr(const_cast<S2V*>(src2.ptr()));
		if(numparams==0){
#if USE_THRUST_LAUNCHER 
			switch(bf){
				case BF_ADD:      thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_plus<DV,S1V,S2V>()); break;
				case BF_SUBTRACT: thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_minus<DV,S1V,S2V>()); break;
				case BF_MULT:     thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_multiplies<DV,S1V,S2V>()); break;
				case BF_DIV:      thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_divides<DV,S1V,S2V>()); break;
				case BF_MIN:      thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_min<DV,S1V,S2V>()); break;
				case BF_MAX:      thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_max<DV,S1V,S2V>()); break;
				case BF_ATAN2:    thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_atan2<DV,S1V,S2V>()); break;
				case BF_NORM:     thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_norm<DV,S1V,S2V>()); break;
				default: cuvAssert(false);
			}
#else
			dim3 blocks, threads;
			setLinearGridAndThreads(blocks,threads,v.size());
			switch(bf){
				case BF_ADD:      launch_binary_kernel(v,w,bf_plus<DV,S1V,S2V>()); break;
				case BF_SUBTRACT: launch_binary_kernel(v,w,bf_minus<DV,S1V,S2V>()); break;
				case BF_MULT:     launch_binary_kernel(v,w,bf_multiplies<DV,S1V,S2V>()); break;
				case BF_DIV:      launch_binary_kernel(v,w,bf_divides<DV,S1V,S2V>()); break;
				case BF_MIN:      launch_binary_kernel(v,w,bf_min<DV,S1V,S2V>()); break;
				case BF_MAX:      launch_binary_kernel(v,w,bf_max<DV,S1V,S2V>()); break;
				case BF_ATAN2:    launch_binary_kernel(v,w,bf_atan2<DV,S1V,S2V>()); break;
				case BF_NORM:    launch_binary_kernel(v,w,bf_norm<DV,S1V,S2V>()); break;
				default: cuvAssert(false);
			}
#endif
		}else if(numparams==1){
#if USE_THRUST_LAUNCHER
			switch(bf){
				case BF_AXPY:     thrust::transform(s1_ptr, s1_ptr+src1.size(), s2_ptr,  d_ptr, bf_axpy<DV,S1V,S2V>(p)); break;
				case BF_XPBY:     thrust::transform(s1_ptr, s1_ptr+src1.size(), s2_ptr,  d_ptr, bf_xpby<DV,S1V,S2V>(p)); break;
						  /*case BF_XPBY:     cublasSaxpy(v.size(), param, (float*)w.ptr(), 1, (float*)v.ptr(), 1) ; break;*/
				default: cuvAssert(false);
			}
#else
			dim3 blocks, threads;
			setLinearGridAndThreads(blocks,threads,v.size());
			switch(bf){
				case BF_AXPY:     launch_binary_kernel(dst,src1,src2,bf_axpy<DV,S1V,S2V>(p)); break;
				case BF_XPBY:     launch_binary_kernel(dst,src1,src2,bf_xpby<DV,S1V,S2V>(p)); break;
				default: cuvAssert(false);
			}
#endif
		}else if(numparams==2){
#if USE_THRUST_LAUNCHER
			switch(bf){
				case BF_AXPBY:     thrust::transform(s1_ptr, s1_ptr+src1.size(), s2_ptr,  d_ptr, bf_axpby<DV,S1V,S2V>(p,p2)); break;
				default: cuvAssert(false);
			}
#else
			dim3 blocks, threads;
			setLinearGridAndThreads(blocks,threads,v.size());
			switch(bf){
				case BF_AXPBY:     launch_binary_kernel(dst,src1,src2,bf_axpby<DV,S1V,S2V>(p,p2)); break;
				default: cuvAssert(false);
			}
#endif
		}
		cuvSafeCall(cudaThreadSynchronize());
	}
};



/*
 * Reductions
 */
template<class __tensor_type>
bool
has_inf(const __tensor_type& v){
	typedef typename __tensor_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __tensor_type::memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<value_type*>(v.ptr()));
	uf_is_inf<value_type> uo;
	return  thrust::any_of(v_ptr, v_ptr+v.size(), uo);
}
template<class __tensor_type>
bool
has_nan(const __tensor_type& v){
	typedef typename __tensor_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __tensor_type::memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<value_type*>(v.ptr()));
	uf_is_nan<value_type> uo;
	return  thrust::any_of(v_ptr, v_ptr+v.size(), uo);
}
template<class __tensor_type>
float
norm2(const __tensor_type& v){
	typedef typename __tensor_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __tensor_type::memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<value_type*>(v.ptr()));
	float init=0;
	return  std::sqrt( thrust::transform_reduce(v_ptr, v_ptr+v.size(), uf_square<float,value_type>(), init, bf_plus<float,float,value_type>()) );
}
template<class __tensor_type>
float
norm1(const __tensor_type& v){
	typedef typename __tensor_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __tensor_type::memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<value_type*>(v.ptr()));
	float init=0;
	uf_abs<float,value_type> unary_op;
	bf_plus<float,float,value_type> binary_op;
	return   thrust::transform_reduce(v_ptr, v_ptr+v.size(), unary_op, init, binary_op);
}
template<class __tensor_type>
float
sum(const __tensor_type& v){
	typedef typename __tensor_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __tensor_type::memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<value_type*>(v.ptr()));
	float init=0.0;
	return   thrust::reduce(v_ptr, v_ptr+v.size(), init, bf_plus<float,float,value_type>());
}
template<class __tensor_type>
float
maximum(const __tensor_type& v){
	typedef typename __tensor_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __tensor_type::memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<value_type*>(v.ptr()));
	float init=-INT_MAX;
	return   thrust::reduce(v_ptr, v_ptr+v.size(), init, bf_max<float,float,value_type>());
}
template<class __tensor_type>
float
minimum(const __tensor_type& v){
	typedef typename __tensor_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __tensor_type::memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<value_type*>(v.ptr()));
	float init=INT_MAX;
	return   thrust::reduce(v_ptr, v_ptr+v.size(), init, bf_min<float,float,value_type>());
}
template<class __tensor_type>
float
mean(const __tensor_type& v){
	return   sum(v) / (float)v.size();
}
template<class __tensor_type>
float
var(const __tensor_type& v){
	typedef typename __tensor_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __tensor_type::memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<value_type*>(v.ptr()));
	float init=0;
	float m = mean(v);
	return   thrust::transform_reduce(v_ptr, v_ptr+v.size(), 
			make_bind2nd(bf_squared_diff<float,value_type,float>(),m),  // result, tensor-type, mean-type
			init, bf_plus<float,float,float>()) / (float)v.size();
}
template<class __tensor_type>
typename __tensor_type::index_type
arg_max(const __tensor_type& v){
	typedef typename __tensor_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __tensor_type::memory_space_type>::ptr_type ptr_type;
	ptr_type begin(const_cast<value_type*>(v.ptr()));
	ptr_type elem = thrust::max_element(begin, begin	+v.size());
	return thrust::distance(begin,elem);
}
template<class __tensor_type>
typename __tensor_type::index_type
arg_min(const __tensor_type& v){
	typedef typename __tensor_type::value_type value_type;
	typedef typename memspace_cuv2thrustptr<value_type,typename __tensor_type::memory_space_type>::ptr_type ptr_type;
	ptr_type begin(const_cast<value_type*>(v.ptr()));
	ptr_type elem = thrust::min_element(begin, begin	+v.size());
	return thrust::distance(begin,elem);
}

/*
 * Template instantiations
 */

#include "tensor_ops_inst.hxx"

} // cuv

