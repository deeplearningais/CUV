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





#include <iostream>
#include <boost/any.hpp>
#include <cuv/basics/dia_matrix.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/tools/texture.h>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/to_tuple.hpp>


using namespace std;

// stuff from NVIDIA SDK
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define small_grid_thread_id(void) ((__umul24(blockDim.x, blockIdx.x) + threadIdx.x))
#define large_grid_thread_id(void) ((__umul24(blockDim.x,blockIdx.x + __umul24(blockIdx.y,gridDim.x)) + threadIdx.x))
#define large_grid_thread_num(void) ((__umul24(blockDim.x,gridDim.x + __umul24(blockDim.y,gridDim.y))))

#define MAX_NUM_IMGS_AT_ONCE 14
#define SEQ_ROW_FACT         1
#define SPMM_BLOCK_SIZE      256


namespace cuv{
	namespace spmv_impl{
		/*
		 *  For a given number of blocks, return a 2D grid large enough to contain them
		 *  FROM NVIDIA SDK
		 */
		dim3 make_large_grid(const unsigned int num_blocks){
			if (num_blocks <= 65535){
				return dim3(num_blocks);
			} else {
				unsigned int side = (unsigned int) ceil(sqrt((double)num_blocks));
				return dim3(side,side);
			}
		}

		dim3 make_large_grid(const unsigned int num_threads, const unsigned int blocksize){
			const unsigned int num_blocks = DIVIDE_INTO(num_threads, blocksize);
			if (num_blocks <= 65535){
				//fits in a 1D grid
				return dim3(num_blocks);
			} else {
				//2D grid is required
				const unsigned int side = (unsigned int) ceil(sqrt((double)num_blocks));
				return dim3(side,side);
			}
		}

// this file is generated using a perl-script from spmv_kernel.cuh
#include "spmv_dia_kernel_inst.cuh"

		template <typename value_type, typename index_type>
			void spmv_dia_device(const dia_matrix<value_type,dev_memory_space,index_type>& A, 
					const tensor<value_type,dev_memory_space>& v, 
					tensor<value_type,dev_memory_space>& dst, 
					char transA,
					const value_type& factAv,
					const value_type& factC)
			{
				const unsigned int toff = bind_x(v.ptr(), v.size());
				spmm_device_dia_dispatch(A,v,dst,transA,factAv,factC,toff);
				cuvSafeCall(cudaThreadSynchronize());
				unbind_x(v.ptr());
			}


		/*template <bool transA, typename value_type, typename index_type>*/
		/*    void spmv_dia_tex_device(const dia_matrix<value_type,dev_memory_space,index_type>& A, */
		/*            const tensor<value_type,dev_memory_space>& v, */
		/*            tensor<value_type,dev_memory_space>& dst)*/
		/*    {*/
		/*        const unsigned int BLOCK_SIZE = 256;*/
		/*        const dim3 grid = make_large_grid(A.shape()[0],BLOCK_SIZE);*/

		/*        cuvAssert(A.num_dia() < BLOCK_SIZE); // kernel doesn't handle larger numbers of diagonals*/

		/*        bind_x(v.ptr());*/

		/*        if(!transA){*/
		/*            const unsigned int BLOCK_SIZE = 256;*/
		/*            const dim3 grid = make_large_grid(A.shape()[0],BLOCK_SIZE);*/
		/*            cuvAssert(A.num_dia() < BLOCK_SIZE); // kernel doesn't handle larger numbers of diagonals*/
		/*            spmv_dia_kernel<value_type, index_type, BLOCK_SIZE, true> <<<grid, BLOCK_SIZE>>> (A.shape()[0], A.shape()[1],  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec().ptr(), v.ptr(), dst.ptr());*/
		/*        }else{*/
		/*            const unsigned int BLOCK_SIZE = 256;*/
		/*            const dim3 grid = make_large_grid(A.shape()[1],BLOCK_SIZE);*/
		/*            cuvAssert(A.num_dia() < BLOCK_SIZE); // kernel doesn't handle larger numbers of diagonals*/
		/*            spmv_dia_kernel_trans<value_type, index_type, BLOCK_SIZE, true> <<<grid, BLOCK_SIZE>>> (A.shape()[0], A.shape()[1],  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec().ptr(), v.ptr(), dst.ptr());*/
		/*        }*/

		/*        unbind_x(v.ptr());*/
		/*    }*/
		template<class value_type>
			void spmv(tensor<value_type,dev_memory_space>& dst, const dia_matrix<value_type,dev_memory_space>& A, const tensor<value_type,dev_memory_space>& v, char transA, const float& factAv, const float& factC){
				// TODO: find a good assert
				/*if(transA=='t'){*/
					/*cuvAssert(A.shape()[1] == dst.size());*/
				/*}else{*/
					/*cuvAssert(A.shape()[0] == dst.size());*/
				/*}*/
				spmv_dia_device(A,v,dst,transA,factAv,factC);
			}


		/****************************************************************
		 *  Host Code
		 ****************************************************************/
		template<class value_type, class index_type>
			void spmv(tensor<value_type,host_memory_space>& dst, const dia_matrix<value_type,host_memory_space,index_type>& A, const tensor<value_type,host_memory_space>& v, char transA, const float& factAv, const float& factC){
				const tensor<int,host_memory_space>& offsets = A.get_offsets();
				const int num_diags             = A.num_dia();
				const int A_h                   = A.h();
				const int A_w                   = A.w();
				const int A_stride              = A.stride();
				index_type max_dst = ((transA=='t') ? A_w : A_h);
				if(factC==0.f)
					for(int i=0;i<max_dst;i++) dst[i]=0;
				else
					for(int i=0;i<max_dst;i++) dst[i]=dst[i] * factC;
				const int rf = A.row_fact();
				if(transA == 't'){
					cuvAssert(A_h == v.size());
					cuvAssert(A_w == dst.size());
					for(index_type i = 0; i < num_diags; i++){
						const int k = offsets[i];  //diagonal offset

						const index_type i_start =  1 * std::max((int)0, k);
						const index_type j_start = rf * std::max((int)0,-k); // the matrix is now _wider_ than high --> stretch columns!

						//number of elements to process
						const index_type N = std::min((A_h - j_start)/rf, A_w - i_start);

						const value_type * d_ = A.vec().ptr() + i*A_stride + j_start;
						const value_type * x_ = v.ptr() + j_start;
						value_type * y_ = dst.ptr() + i_start;

						for(index_type n = 0; n < N; n++,y_++){
							for(int k=0;k<rf;k++,x_++,d_++)
								*y_ += factAv * *d_ * *x_;
						}
					}
				}else{
					cuvAssert(A_w == v.size());
					cuvAssert(A_h == dst.size());
					for(index_type i = 0; i < num_diags; i++){
						const int k = offsets[i];  //diagonal offset

						const index_type i_start = rf*std::max((int)0,-k);
						const index_type j_start =  1*std::max((int)0, k);

						//number of elements to process
						const index_type N = std::min(A_h - i_start, rf*(A_w - j_start));

						const value_type * d_ = A.vec().ptr() + i*A_stride + i_start;
						const value_type * x_ = v.ptr() + j_start;
						value_type * y_ = dst.ptr() + i_start;

						for(index_type n = 0; n < N; n++){
							*y_++ += factAv * *d_++ * x_[n/rf];
						}
					}
				}
			}
	}

	template<>
		void prod(tensor<float,host_memory_space,column_major>& dst,
				  const dia_matrix<float,host_memory_space>&                  A,
				  const tensor<float,host_memory_space,column_major>&   B,
				  char transA,
				  char transB,
				  const float& factAB,
				  const float& factC){
                        cuvAssert(dst.shape().size()==2);
                        cuvAssert(B.shape().size()==2);
			cuvAssert(transB == 'n');
			cuvAssert(dst.shape()[1] == B.shape()[1]);
			for(int i=0;i<dst.shape()[1];i++){
				tensor<float,host_memory_space> dst_v(indices[index_range(0,dst.shape()[0])], dst.ptr()+i*dst.shape()[0]);
				tensor<float,host_memory_space> src_v(indices[index_range(0,B.shape()[0])],   B.ptr()+i*B.shape()[0]);
				spmv(dst_v,A,src_v,transA,factAB,factC);
			}
		}
	template<>
		void prod(tensor<float,dev_memory_space,column_major>& dst,
				  const dia_matrix<float,dev_memory_space>&                  A,
				  const tensor<float,dev_memory_space,column_major>&   B,
				  char transA,
				  char transB,
				  const float& factAB,
				  const float& factC){
                        cuvAssert(dst.shape().size()==2);
                        cuvAssert(B.shape().size()==2);
			cuvAssert(transB == 'n');
			cuvAssert(dst.shape()[1] == B.shape()[1]);
			cuvAssert(dst.ptr());
			cuvAssert(A.vec_ptr());
			cuvAssert(B.ptr());
			if(transA=='t'){
				cuvAssert(A.w() == dst.shape()[0]);
			}else{
				cuvAssert(A.h() == dst.shape()[0]);
			}
			const int num_at_same_time = min(MAX_NUM_IMGS_AT_ONCE, B.shape()[1]);
			for(int i=0; i<dst.shape()[1]; i += num_at_same_time){
				tensor<float,dev_memory_space> dst_v(indices[index_range(0,dst.shape()[0] * min(dst.shape()[1]-i,num_at_same_time))], dst.ptr()+i*dst.shape()[0]);
				tensor<float,dev_memory_space> src_v(indices[index_range(0,B.shape()[0]* min(B.shape()[1]-i,  num_at_same_time))], B.ptr()+i*B.shape()[0]);
				spmv(dst_v,A,src_v,transA,factAB,factC);
			}
		}
      template<class __value_type, class __memory_space_type>
              void spmv(tensor<__value_type, __memory_space_type>& dst, const dia_matrix<__value_type, __memory_space_type>& A, const tensor<__value_type, __memory_space_type>& v,const  char transA, const float& factAv, const float& factC){
			spmv_impl::spmv(dst,A,v,transA,factAv,factC);
		}
	template void spmv<float,host_memory_space>(tensor<float,host_memory_space>&dst, const dia_matrix<float,host_memory_space>& A, const tensor<float,host_memory_space>& v, char, const float&, const float&);
	template void spmv<float,dev_memory_space>(tensor<float,dev_memory_space>&dst, const dia_matrix<float,dev_memory_space>& A, const tensor<float,dev_memory_space>& v, char, const float&, const float&);
}
