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
#include <thrust/count.h>
#include <thrust/extrema.h>

#include <cuv/tools/cuv_general.hpp>

#include <cuv/basics/tensor.hpp>
#include <cuv/tensor_ops/functors.hpp>

#include <cuv/tensor_ops/tensor_ops.hpp>

/**
 * @defgroup internal functors
 * @{
 */


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
/**
 * apply a unary functor pointwise
 */
template<class unary_functor, class value_type, class index_type>
__global__
void unary_functor_kernel(value_type* dst, value_type* src n, unary_functor uf){
	const unsigned int idx = __mul24(blockIdx.x , blockDim.x) + threadIdx.x;
	const unsigned int off = __mul24(blockDim.x , gridDim.x);
	for (unsigned int i = idx; i < n; i += off)
		dst[i] = uf(src[i]);
}

/**
 * apply a binary functor pointwise
 */
template<class binary_functor, class value_type, class value_type2, class index_type>
__global__
void binary_functor_kernel(value_type* dst, value_type* src, value_type2* src2 n, binary_functor bf){
	const unsigned int idx = __mul24(blockIdx.x , blockDim.x) + threadIdx.x;
	const unsigned int off = __mul24(blockDim.x , gridDim.x);
	for (unsigned int i = idx; i < n; i += off)
		dst[i] = bf(src[i],src2[i]);
}

/**
 * determine the size of a grid/block for pointwise kernels with little/no memory consumption
 */
void setLinearGridAndThreads(dim3& blocks, dim3& threads, size_t len, int threads_per_block=512){
	const int padded_len=(int)ceil((float)len/threads_per_block)*threads_per_block;
	blocks = dim3(min(512,padded_len/threads_per_block),1,1);
	threads = dim3(threads_per_block,1,1);
}
#endif

/**
 * Launch a unary kernel on device
 * 
 * @param dst where result is written to
 * @param src the   data which is transformed
 * @param uf  the unary functor applied
 * @param mask  if given, dst[i]=uf(src[i]) is only applied when mask[i]!=0
 */
template<class unary_functor, class V1, class V2>
void launch_unary_kernel(
   cuv::tensor<V1,dev_memory_space>& dst,
   const cuv::tensor<V2,dev_memory_space>& src, 
	 unary_functor uf,
	 const tensor<unsigned char,dev_memory_space>* mask
	 ){
	 cuvAssert(dst.ptr());
	 cuvAssert(src.ptr());
	 cuvAssert(dst.size() == src.size());

#if ! USE_THRUST_LAUNCHER
	 dim3 blocks, threads;
	 setLinearGridAndThreads(blocks,threads,dst.size());
	 unary_functor_kernel<<<blocks,threads>>>(dst.ptr(),src.ptr(),dst.size(),uf); //     180 ms
#else
	 if(!mask){
		 thrust::device_ptr<V1> dst_ptr(dst.ptr());
		 thrust::device_ptr<V2> src_ptr(const_cast<V2*>(src.ptr()));
		 thrust::transform(src_ptr,src_ptr+src.size(),dst_ptr,uf);
	 }else{
		 cuvAssert(mask->ptr());
		 thrust::device_ptr<V1> dst_ptr(dst.ptr());
		 thrust::device_ptr<V2> src_ptr(const_cast<V2*>(src.ptr()));
		 thrust::device_ptr<unsigned char> mask_ptr(const_cast<unsigned char*>(mask->ptr()));
		 thrust::transform_if(src_ptr,src_ptr+src.size(),mask_ptr,dst_ptr,uf,make_bind2nd(thrust::not_equal_to<unsigned char>(),(unsigned char)0));
	 }
#endif

	 cuvSafeCall(cudaThreadSynchronize());
}

/**
 * @overload
 *
 * Launch unary kernel on device
 */
template<class unary_functor, class V1, class V2>
void launch_unary_kernel(
   cuv::tensor<V1,host_memory_space>& dst,
   const cuv::tensor<V2,host_memory_space>& src, 
	 unary_functor uf, const tensor<unsigned char, host_memory_space>* mask){
	 cuvAssert(src.ptr());
	 cuvAssert(dst.ptr());
	 cuvAssert(dst.size() == src.size());
	 V1* dst_ptr = dst.ptr();
	 const V2* src_ptr = src.ptr();
     size_t size = dst.size();
	 if(!mask)
		 for(size_t i=0;i<size;i++)
			 *dst_ptr++ = uf( *src_ptr++ );
	 else{
		 cuvAssert(mask->ptr());
		 const unsigned char* mask_ptr = mask->ptr();
		 for(size_t i=0;i<size;i++,src_ptr++)
			 *dst_ptr++ = *mask_ptr++ ? uf( *src_ptr ) : *src_ptr;
	 }
}

/**
 * Launch binary kernel on device
 * 
 * @param v destination and first parameter of bf
 * @param w second parameter of bf
 * @param bf the binary functor to be applied
 */
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

/**
 * @overload
 *
 * launch a binary kernel on host
 */
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
     const size_t size = dst.size();
	 for(size_t i=0;i<size;i++)
	   *dst_ptr++ = uf(*dst_ptr,*src_ptr++);
}

namespace cuv{
	
/**
 * Nullary Functor
 *
 * generates numbers, e.g. a sequence
 *
 * @param v the destination tensor
 * @param nf the functor to be applied
 */

template<class __value_type, class __memory_space_type>
void
apply_0ary_functor(tensor<__value_type, __memory_space_type>& v, const NullaryFunctor& nf){
	 cuvAssert(v.ptr());
	 typedef typename memspace_cuv2thrustptr<__value_type,__memory_space_type>::ptr_type ptr_type;
	 ptr_type dst_ptr(v.ptr());
	 switch(nf){
		 case NF_SEQ:
			 thrust::sequence(dst_ptr,dst_ptr+v.size());break;
		 default:
			 cuvAssert(false);
	 }
	 cuvSafeCall(cudaThreadSynchronize());
}

/**
 * @overload
 *
 * for null-ary functors which are generated by taking a unary functor and
 * fixing  the argument to a constant
 *
 * @param v where data is written to
 * @param nf how data is generated (e.g. NF_FILL)
 * @param param  the parameter of the unary functor to be fixed
 */
template<class V1, class M>
void
apply_0ary_functor(tensor<V1, M>& v, const NullaryFunctor& nf, const V1& param){
	 cuvAssert(v.ptr());

	 typedef typename memspace_cuv2thrustptr<V1,M >::ptr_type ptr_type;
	 ptr_type dst_ptr(v.ptr());
	 switch(nf){
		 case NF_FILL:
			 thrust::fill(dst_ptr,dst_ptr + v.size(), (V1)param); break;
		 default:
			 cuvAssert(false);
	 }
	 cuvSafeCall(cudaThreadSynchronize());
}


namespace detail{
	// **********************************
	//       Scalar Functor
	// **********************************
	template<class V1, class V2, class M, class S1, class S2>
	void apply_scalar_functor(tensor<V1, M>& dst, const tensor<V2, M>& src, const ScalarFunctor& sf, const int& numparams, const tensor<unsigned char,M>* mask, const S1& p, const S2& p2){
		cuvAssert(equal_shape(dst,src));

		typedef typename memspace_cuv2thrustptr<V1, M>::ptr_type ptr_type1;
		typedef typename memspace_cuv2thrustptr<V2, M>::ptr_type ptr_type2;
		ptr_type1 d_ptr(dst.ptr());
		ptr_type2 s_ptr(const_cast<V2*>(src.ptr()));
		if(numparams==2){
			switch(sf){
				case SF_AXPB:      launch_unary_kernel(dst,src,make_bind2nd3rd(tf_axpb <V1>(),p,p2),mask); break;
				case SF_TANH:      launch_unary_kernel(dst,src,make_bind2nd3rd(tf_tanh <V1>(),p,p2),mask); break;
				case SF_DTANH:     launch_unary_kernel(dst,src,make_bind2nd3rd(tf_dtanh<V1>(),p,p2),mask); break;
				default:
						   cout << "No suitable two-parameter scalar functor was found." << endl;
						   cuvAssert(false);
			}
		}
		else if(numparams==1){
			switch(sf){
				case SF_POW:            launch_unary_kernel(dst,src,make_bind2nd(bf_pow<V1,V2,S1>(),p),mask); break;
				case SF_DPOW:           launch_unary_kernel(dst,src,make_bind2nd(bf_dpow<V1,V2,S1>(),p),mask); break;
				case SF_SIGM:           launch_unary_kernel(dst,src,make_bind2nd(bf_sigm_temp<V1,V2>(),p),mask); break;
				case SF_ADD:            launch_unary_kernel(dst,src,make_bind2nd(thrust::plus<V1>(),p),mask); break;
				case SF_MULT:           launch_unary_kernel(dst,src,make_bind2nd(thrust::multiplies<V1>(),p),mask); break;
				case SF_DIV:            launch_unary_kernel(dst,src,make_bind2nd(thrust::divides<V1>(),p),mask); break;
				case SF_RDIV:           launch_unary_kernel(dst,src,make_bind1st(thrust::divides<V1>(),p),mask); break;
				case SF_SUBTRACT:       launch_unary_kernel(dst,src,make_bind2nd(thrust::minus<V1>(),p),mask); break;
				case SF_RSUB:           launch_unary_kernel(dst,src,make_bind1st(thrust::minus<V1>(),p),mask); break;
				case SF_LOGADDEXP:      launch_unary_kernel(dst,src,make_bind1st(bf_logaddexp<V1>(),p),mask); break;
                 case SF_LOGADDEXP_GRAD: launch_unary_kernel(dst,src,make_bind1st(bf_logaddexp_grad<V1>(),p),mask); break;                
				case SF_MIN:            launch_unary_kernel(dst,src,make_bind2nd(bf_min<V1,V2,S1>(),p),mask); break;
				case SF_MAX:            launch_unary_kernel(dst,src,make_bind2nd(bf_max<V1,V2,S1>(),p),mask); break;
				case SF_ROBUST_ABS:     launch_unary_kernel(dst,src,make_bind2nd(bf_robust_abs<V1,V2,S1>(),p),mask); break;
				case SF_DROBUST_ABS:    launch_unary_kernel(dst,src,make_bind2nd(bf_drobust_abs<V1,V2,S1>(),p),mask); break;
				case SF_RECT:           launch_unary_kernel(dst,src,make_bind2nd(bf_rect<V1,V2,S1>(),p),mask); break;
				case SF_DRECT:          launch_unary_kernel(dst,src,make_bind2nd(bf_drect<V1,V2,S1>(),p),mask); break;
				case SF_EQ:             launch_unary_kernel(dst,src,make_bind2nd(thrust::equal_to<V2>(),p),mask); break;
				case SF_LT:             launch_unary_kernel(dst,src,make_bind2nd(thrust::less<V2>(),p),mask); break;
				case SF_GT:             launch_unary_kernel(dst,src,make_bind2nd(thrust::greater<V2>(),p),mask); break;
				case SF_LEQ:            launch_unary_kernel(dst,src,make_bind2nd(thrust::less_equal<V2>(),p),mask); break;
				case SF_GEQ:            launch_unary_kernel(dst,src,make_bind2nd(thrust::greater_equal<V2>(),p),mask); break;
				case SF_BERNOULLI_KL:   launch_unary_kernel(dst,src,make_bind1st(bf_bernoulli_kl<V1,V2,S1>(),p),mask); break;
				case SF_DBERNOULLI_KL:  launch_unary_kernel(dst,src,make_bind1st(bf_dbernoulli_kl<V1,V2,S1>(),p),mask); break;
				default:
						   cout << "No suitable one-parameter scalar functor was found." << endl;
						   cuvAssert(false);
			}
		}
		else if(numparams==0){
			switch(sf){
				case SF_EXP:        launch_unary_kernel(dst,src, uf_exp<V1,V2>(),mask); break;
				case SF_SIN:        launch_unary_kernel(dst,src, uf_sin<V1,V2>(),mask); break;
				case SF_COS:        launch_unary_kernel(dst,src, uf_cos<V1,V2>(),mask); break;
				case SF_LOG:        launch_unary_kernel(dst,src, uf_log<V1,V2>(),mask); break;
				case SF_SIGN:       launch_unary_kernel(dst,src, uf_sign<V1,V2>(),mask); break;
				case SF_SIGM:       launch_unary_kernel(dst,src, uf_sigm<V1,V2>(),mask); break;
				case SF_DSIGM:      launch_unary_kernel(dst,src, uf_dsigm<V1,V2>(),mask); break;
				case SF_TANH:       launch_unary_kernel(dst,src, uf_tanh<V1,V2>(),mask); break;
				case SF_DTANH:      launch_unary_kernel(dst,src, uf_dtanh<V1,V2>(),mask); break;
				case SF_SQUARE:     launch_unary_kernel(dst,src, uf_square<V1,V2>(),mask); break;
				case SF_SUBLIN:     launch_unary_kernel(dst,src, uf_sublin<V1,V2>(),mask); break;
				case SF_ENERG:      launch_unary_kernel(dst,src, uf_energ<V1,V2>(),mask); break;
				case SF_INV:        launch_unary_kernel(dst,src, uf_inv<V1,V2>(),mask); break;
				case SF_SQRT:       launch_unary_kernel(dst,src, uf_sqrt<V1,V2>(),mask); break;
				case SF_SMAX:       launch_unary_kernel(dst,src, uf_smax<V1,V2>(),mask); break;
				case SF_NEGATE:     launch_unary_kernel(dst,src, thrust::negate<V1>(),mask); break;
				case SF_ABS:        launch_unary_kernel(dst,src, uf_abs<V1,V2>(),mask); break;
				case SF_POSLIN:     launch_unary_kernel(dst,src, uf_poslin<V1,V2>(),mask); break;
				case SF_COPY:       launch_unary_kernel(dst,src, uf_identity<V1,V2>(),mask); break; //thrust::copy(s_ptr, s_ptr+src.size(), d_ptr); break;
				case SF_LOG1P:      launch_unary_kernel(dst,src, uf_log1p<V1,V2>(),mask); break;
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
	template<class V1, class V2, class V3, class M, class S1, class S2>
	  void apply_binary_functor(tensor<V1, M>& dst,const tensor<V2, M>& src1, const tensor<V3, M>& src2, const BinaryFunctor& bf, const int& numparams, const S1& p, const S2& p2){
        bool src1_agrees = equal_shape(dst,src1);
        bool src2_agrees = equal_shape(dst,src2);
        cuvAssert(src1_agrees || src2_agrees);

        

       
        

		typedef typename memspace_cuv2thrustptr<V1, M>::ptr_type ptr_type1;
		typedef typename memspace_cuv2thrustptr<V2,M>::ptr_type ptr_type2;
		typedef typename memspace_cuv2thrustptr<V3,M>::ptr_type ptr_type3;
		ptr_type1 d_ptr(dst.ptr());
		ptr_type2 s1_ptr(const_cast<V2*>(src1.ptr()));
		ptr_type2 s2_ptr(const_cast<V3*>(src2.ptr()));
        if(numparams==0){
            if(!src1_agrees && src1.size() == 1){
                switch(bf){
                    case BF_EQ:       apply_scalar_functor(dst, src2, SF_EQ,  src1[0]); break;
                    case BF_ADD:      apply_scalar_functor(dst, src2, SF_ADD,  src1[0]);break;
                    case BF_SUBTRACT: apply_scalar_functor(dst, src2, SF_SUBTRACT,  src1[0]);break;
                    case BF_MULT:     apply_scalar_functor(dst, src2, SF_MULT,  src1[0]);break;
                    case BF_DIV:      apply_scalar_functor(dst, src2, SF_DIV,  src1[0]);break;
                    case BF_MIN:      apply_scalar_functor(dst, src2, SF_MIN,  src1[0]);break;
                    case BF_MAX:      apply_scalar_functor(dst, src2, SF_MAX,  src1[0]);break;
                    case BF_LOGADDEXP:     apply_scalar_functor(dst, src2, SF_LOGADDEXP,  src1[0]); break;
                    case BF_LOGADDEXP_GRAD:     apply_scalar_functor(dst, src2, SF_LOGADDEXP_GRAD,  src1[0]); break;                    
                    case BF_BERNOULLI_KL:      apply_scalar_functor(dst, src2, SF_BERNOULLI_KL,  src1[0]); break;
                    case BF_DBERNOULLI_KL:     apply_scalar_functor(dst, src2, SF_DBERNOULLI_KL,  src1[0]); break;
                    default: throw std::runtime_error("supplied binary functor broadcast not implemented");
                }
                return;
            }
            else if(!src2_agrees && src2.size() == 1){
                switch(bf){
                    case BF_EQ:       apply_scalar_functor(dst, src1, SF_EQ,  src2[0]); break;
                    case BF_ADD:      apply_scalar_functor(dst, src1, SF_ADD,  src2[0]);break;
                    case BF_SUBTRACT: apply_scalar_functor(dst, src1, SF_SUBTRACT,  src2[0]);break;
                    case BF_MULT:     apply_scalar_functor(dst, src1, SF_MULT,  src2[0]);break;
                    case BF_DIV:      apply_scalar_functor(dst, src1, SF_DIV,  src2[0]);break;
                    case BF_MIN:      apply_scalar_functor(dst, src1, SF_MIN,  src2[0]);break;
                    case BF_MAX:      apply_scalar_functor(dst, src1, SF_MAX,  src2[0]);break;
                    case BF_LOGADDEXP:     apply_scalar_functor(dst, src1, SF_LOGADDEXP,  src2[0]); break;
                    case BF_LOGADDEXP_GRAD:     apply_scalar_functor(dst, src1, SF_LOGADDEXP_GRAD,  src2[0]); break;
                    case BF_BERNOULLI_KL:      apply_scalar_functor(dst, src1, SF_BERNOULLI_KL,  src2[0]); break;
                    case BF_DBERNOULLI_KL:     apply_scalar_functor(dst, src1, SF_DBERNOULLI_KL,  src2[0]); break;
                    default: throw std::runtime_error("supplied binary functor broadcast not implemented");
                }
                return;
            }else if(!src1_agrees || !src2_agrees){
                throw std::runtime_error("For binary functor, all dimensions must agree OR one of the argument must be a scalar vector.");
            }
#if USE_THRUST_LAUNCHER 
            switch(bf){
                case BF_1ST:      thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_1st<V1,V2,V3>()); break;
                case BF_2ND:      thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_2nd<V1,V2,V3>()); break;
                case BF_EQ:       thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_equals<V1,V2,V3>()); break;
                case BF_AND:      thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_and<V1,V2,V3>()); break;
                case BF_OR :      thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_or<V1,V2,V3>()); break;
                case BF_ADD:      thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_plus<V1,V2,V3>()); break;
                case BF_SUBTRACT: thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_minus<V1,V2,V3>()); break;
                case BF_MULT:     thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_multiplies<V1,V2,V3>()); break;
                case BF_DIV:      thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_divides<V1,V2,V3>()); break;
                case BF_MIN:      thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_min<V1,V2,V3>()); break;
                case BF_MAX:      thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_max<V1,V2,V3>()); break;
                case BF_ATAN2:    thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_atan2<V1,V2,V3>()); break;
                case BF_NORM:     thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_norm<V1,V2,V3>()); break;
                case BF_LOGADDEXP:     thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_logaddexp<V1>()); break;
                case BF_LOGADDEXP_GRAD:     thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_logaddexp_grad<V1>()); break;                
                case BF_LOGCE_OF_LOGISTIC:     thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_logce_of_logistic<V1,V2,V3>()); break;
                case BF_BERNOULLI_KL:      thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_bernoulli_kl<V1,V2,V3>()); break;
                case BF_DBERNOULLI_KL:     thrust::transform(s1_ptr, s1_ptr+dst.size(), s2_ptr, d_ptr, bf_dbernoulli_kl<V1,V2,V3>()); break;
                default: cuvAssert(false);
            }
#else
            dim3 blocks, threads;
            setLinearGridAndThreads(blocks,threads,v.size());
            switch(bf){
                case BF_AND:      launch_binary_kernel(v,w,bf_and<V1,V2,V3>()); break;
                case BF_OR :      launch_binary_kernel(v,w,bf_or<V1,V2,V3>()); break;
                case BF_ADD:      launch_binary_kernel(v,w,bf_plus<V1,V2,V3>()); break;
                case BF_SUBTRACT: launch_binary_kernel(v,w,bf_minus<V1,V2,V3>()); break;
                case BF_MULT:     launch_binary_kernel(v,w,bf_multiplies<V1,V2,V3>()); break;
                case BF_DIV:      launch_binary_kernel(v,w,bf_divides<V1,V2,V3>()); break;
                case BF_MIN:      launch_binary_kernel(v,w,bf_min<V1,V2,V3>()); break;
                case BF_MAX:      launch_binary_kernel(v,w,bf_max<V1,V2,V3>()); break;
                case BF_POW:      launch_binary_kernel(v,w,bf_pow<V1,V2,V3>()); break;
                case BF_DPOW:     launch_binary_kernel(v,w,bf_dpow<V1,V2,V3>()); break;
                case BF_ATAN2:    launch_binary_kernel(v,w,bf_atan2<V1,V2,V3>()); break;
                case BF_NORM:    launch_binary_kernel(v,w,bf_norm<V1,V2,V3>()); break;
                case BF_LOGADDEXP:          launch_binary_kernel(v,w,bf_logaddexp<V1>()); break;
                case BF_LOGADDEXP_GRAD:     launch_binary_kernel(v,w,bf_logaddexp_grad<V1>()); break;                
                case BF_LOGCE_OF_LOGISTIC:  launch_binary_kernel(v,w,bf_logce_of_logistic<V1,V2,V3>()); break;
                default: cuvAssert(false);
            }
#endif
        }else if(numparams==1){
            if(!src1_agrees && src1.size() == 1){
                switch(bf){
                    case BF_AXPY: apply_scalar_functor(dst, src2, SF_ADD,  p * src1[0]); break;
                    case BF_XPBY: apply_scalar_functor(dst, src2, SF_AXPB, p,  src1[0]); break;
                    default: throw std::runtime_error("supplied binary functor broadcast not implemented");
                }
                return;
            }
            else if(!src2_agrees && src2.size() == 1){
                switch(bf){
                    case BF_AXPY: apply_scalar_functor(dst, src1, SF_AXPB, p,  src2[0]); break;
                    case BF_XPBY: apply_scalar_functor(dst, src1, SF_ADD,  p * src2[0]); break;
                    default: throw std::runtime_error("supplied binary functor broadcast not implemented");
                }
                return;
            }else if(!src1_agrees || !src2_agrees){
                throw std::runtime_error("For binary functor, all dimensions must agree OR one of the argument must be a scalar vector.");
            }
#if USE_THRUST_LAUNCHER
			switch(bf){
				case BF_AXPY:     thrust::transform(s1_ptr, s1_ptr+src1.size(), s2_ptr,  d_ptr, bf_axpy<V1,V2,V3>(p)); break;
				case BF_XPBY:     thrust::transform(s1_ptr, s1_ptr+src1.size(), s2_ptr,  d_ptr, bf_xpby<V1,V2,V3>(p)); break;
				case BF_EPSILON_INSENSITIVE_LOSS: thrust::transform(s1_ptr, s1_ptr+src1.size(), s2_ptr,  d_ptr, make_bind3rd(tf_epsilon_insensitive_loss<V1>(),p)); break;
				case BF_DEPSILON_INSENSITIVE_LOSS: thrust::transform(s1_ptr, s1_ptr+src1.size(), s2_ptr,  d_ptr, make_bind3rd(tf_depsilon_insensitive_loss<V1>(),p)); break;
				case BF_HINGE_LOSS: thrust::transform(s1_ptr, s1_ptr+src1.size(), s2_ptr,  d_ptr, make_bind3rd(tf_hinge_loss<V1>(),p)); break;
				case BF_DHINGE_LOSS: thrust::transform(s1_ptr, s1_ptr+src1.size(), s2_ptr,  d_ptr, make_bind3rd(tf_dhinge_loss<V1>(),p)); break;
				case BF_SQHINGE_LOSS: thrust::transform(s1_ptr, s1_ptr+src1.size(), s2_ptr,  d_ptr, make_bind3rd(tf_sqhinge_loss<V1>(),p)); break;
				case BF_DSQHINGE_LOSS: thrust::transform(s1_ptr, s1_ptr+src1.size(), s2_ptr,  d_ptr, make_bind3rd(tf_dsqhinge_loss<V1>(),p)); break;
						  /*case BF_XPBY:     cublasSaxpy(v.size(), param, (float*)w.ptr(), 1, (float*)v.ptr(), 1) ; break;*/
				default: cuvAssert(false);
			}
#else
			dim3 blocks, threads;
			setLinearGridAndThreads(blocks,threads,v.size());
			switch(bf){
				case BF_AXPY:     launch_binary_kernel(dst,src1,src2,bf_axpy<V1,V2,V3>(p)); break;
				case BF_XPBY:     launch_binary_kernel(dst,src1,src2,bf_xpby<V1,V2,V3>(p)); break;
				default: cuvAssert(false);
			}
#endif
		}else if(numparams==2){
            if(!src1_agrees && src1.size() == 1){
                switch(bf){
                    case BF_AXPBY: apply_scalar_functor(dst, src2, SF_AXPB, p2, src1[0] * p); break;
                    default: throw std::runtime_error("supplied binary functor broadcast not implemented");
                }
                return;
            }
            else if(!src2_agrees && src2.size() == 1){
                switch(bf){
                    case BF_AXPBY: apply_scalar_functor(dst, src1, SF_AXPB, p, src2[0] * p2); break;
                    default: throw std::runtime_error("supplied binary functor broadcast not implemented");
                }
                return;
            }else if(!src1_agrees || !src2_agrees){
                throw std::runtime_error("For binary functor, all dimensions must agree OR one of the argument must be a scalar vector.");
            }
#if USE_THRUST_LAUNCHER
			switch(bf){
				case BF_AXPBY:     thrust::transform(s1_ptr, s1_ptr+src1.size(), s2_ptr,  d_ptr, bf_axpby<V1,V2,V3>(p,p2)); break;
				default: cuvAssert(false);
			}
#else
			dim3 blocks, threads;
			setLinearGridAndThreads(blocks,threads,v.size());
			switch(bf){
				case BF_AXPBY:     launch_binary_kernel(dst,src1,src2,bf_axpby<V1,V2,V3>(p,p2)); break;
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
template<class __value_type, class __memory_space_type>
bool
has_inf(const tensor<__value_type, __memory_space_type>& v){
	typedef typename memspace_cuv2thrustptr<__value_type,__memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<__value_type*>(v.ptr()));
	uf_is_inf<__value_type> uo;
	return  thrust::any_of(v_ptr, v_ptr+v.size(), uo);
}
template<class __value_type, class __memory_space_type>
bool
has_nan(const tensor<__value_type, __memory_space_type>& v){
	typedef typename memspace_cuv2thrustptr<__value_type,__memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<__value_type*>(v.ptr()));
	uf_is_nan<__value_type> uo;
	return  thrust::any_of(v_ptr, v_ptr+v.size(), uo);
}
template<class __value_type, class __memory_space_type>
float
norm2(const tensor<__value_type, __memory_space_type>& v){
	typedef typename memspace_cuv2thrustptr<__value_type,__memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<__value_type*>(v.ptr()));
	float init=0;
	return  std::sqrt( thrust::transform_reduce(v_ptr, v_ptr+v.size(), uf_square<float,__value_type>(), init, bf_plus<float,float,__value_type>()) );
}
struct squared_diff{
    template<typename Tuple>
    __host__ __device__
    float operator()(Tuple t){
        return pow(float(thrust::get<0>(t)-thrust::get<1>(t)),2.f);
    }
};
template<class __value_type, class __memory_space_type>
float
diff_norm2(const tensor<__value_type, __memory_space_type>& v, const tensor<__value_type, __memory_space_type>& w){
	typedef typename memspace_cuv2thrustptr<__value_type,__memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<__value_type*>(v.ptr()));
	ptr_type w_ptr(const_cast<__value_type*>(w.ptr()));
	float init=0;
	return  std::sqrt( thrust::transform_reduce(
                thrust::make_zip_iterator(thrust::make_tuple(v_ptr,w_ptr)), 
                thrust::make_zip_iterator(
                    thrust::make_tuple(
                        v_ptr+v.size(),
                        w_ptr+w.size())), 
                squared_diff(),
                init,
                bf_plus<float,float,float>() ));
}
template<class __value_type, class __memory_space_type>
float
norm1(const tensor<__value_type, __memory_space_type>& v){
	typedef typename memspace_cuv2thrustptr<__value_type,__memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<__value_type*>(v.ptr()));
	float init=0;
	uf_abs<float,__value_type> unary_op;
	bf_plus<float,float,__value_type> binary_op;
	return   thrust::transform_reduce(v_ptr, v_ptr+v.size(), unary_op, init, binary_op);
}
template<class __value_type, class __memory_space_type>
float
sum(const tensor<__value_type, __memory_space_type>& v){
	typedef typename memspace_cuv2thrustptr<__value_type,__memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<__value_type*>(v.ptr()));
	float init=0.0;
	return   thrust::reduce(v_ptr, v_ptr+v.size(), init, bf_plus<float,float,__value_type>());
}
template<class __value_type, class __memory_space_type>
unsigned int
count(const tensor<__value_type, __memory_space_type>& v, const __value_type& s){
	typedef typename memspace_cuv2thrustptr<__value_type,__memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<__value_type*>(v.ptr()));
	return   thrust::count(v_ptr, v_ptr+v.size(), s);
}
template<class __value_type, class __memory_space_type>
float
maximum(const tensor<__value_type, __memory_space_type>& v){
	typedef typename memspace_cuv2thrustptr<__value_type,__memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<__value_type*>(v.ptr()));
	float init=-INT_MAX;
	return   thrust::reduce(v_ptr, v_ptr+v.size(), init, bf_max<float,float,__value_type>());
}
template<class __value_type, class __memory_space_type>
float
minimum(const tensor<__value_type, __memory_space_type>& v){
	typedef typename memspace_cuv2thrustptr<__value_type,__memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<__value_type*>(v.ptr()));
	float init=INT_MAX;
	return   thrust::reduce(v_ptr, v_ptr+v.size(), init, bf_min<float,float,__value_type>());
}
template<class __value_type, class __memory_space_type>
float
mean(const tensor<__value_type, __memory_space_type>& v){
	return   sum(v) / (float)v.size();
}
template<class __value_type, class __memory_space_type>
float
var(const tensor<__value_type, __memory_space_type>& v){
	typedef typename memspace_cuv2thrustptr<__value_type,__memory_space_type>::ptr_type ptr_type;
	ptr_type v_ptr(const_cast<__value_type*>(v.ptr()));
	float init=0;
	float m = mean(v);
	return   thrust::transform_reduce(v_ptr, v_ptr+v.size(), 
			make_bind2nd(bf_squared_diff<float,__value_type,float>(),m),  // result, tensor-type, mean-type
			init, bf_plus<float,float,float>()) / (float)v.size();
}
template<class __value_type, class __memory_space_type>
typename tensor<__value_type, __memory_space_type>::index_type
arg_max(const tensor<__value_type, __memory_space_type>& v){
	typedef typename memspace_cuv2thrustptr<__value_type,__memory_space_type>::ptr_type ptr_type;
	ptr_type begin(const_cast<__value_type*>(v.ptr()));
	ptr_type elem = thrust::max_element(begin, begin	+v.size());
	return thrust::distance(begin,elem);
}
template<class __value_type, class __memory_space_type>
typename tensor<__value_type, __memory_space_type>::index_type
arg_min(const tensor<__value_type, __memory_space_type>& v){
	typedef typename memspace_cuv2thrustptr<__value_type,__memory_space_type>::ptr_type ptr_type;
	ptr_type begin(const_cast<__value_type*>(v.ptr()));
	ptr_type elem = thrust::min_element(begin, begin	+v.size());
	return thrust::distance(begin,elem);
}

} // cuv

/**
 * @}
 */

