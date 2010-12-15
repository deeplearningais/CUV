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


#include <stdexcept>
#include <cublas.h>
#include <cblas.h>
#include <stdio.h>

#include <cuv_general.hpp>
#include <3rd_party/CudaConv/nvmatrix.cuh>
#include <thrust/functional.h>
#include <float.h>
#include "matrix_ops.hpp"
#include <limits>
#include "../tools/cuPrintf.cu"

#ifdef __CDT_PARSER__
#define __global__
#define __shared__
#endif

#define CVT_TRANSPOSE(c) \
	(CBLAS_TRANSPOSE)(((c) == 'N' || (c) == 'n') ? CblasNoTrans : \
	 ((c) == 'T' || (c) == 't') ? CblasTrans : \
	 ((c) == 'C' || (c) == 'c') ? CblasConjTrans : \
	 -1)
	/* (mg)the idea is to place the blocks under each other starting at the upper left in the matrix. their threads
	 * add up multiples of their x position (indicated by ty - see above) in shared memory. then we have a 2-dim
	 * array in shared memory that corresponds in size to the block.
	 *
	 * in each block we now have to add up the elements of each row in the shared memory to get the final value. this is done
	 * by logarithmically adding up the elements:
	 * in the first step the second half (in x direction) of the blocks add their values to the first halfs memory locations in
	 * shared memory - then this is repeated for the first half of the threads. a short example for blockDim.y=8 and blockDim.x=1:
	 *
	 * 1st iter(offset=4): a1 a2 a3 a4 | b1 b2 b3 b4
	 *
	 * 2nd iter(offset=2): (a1+b1) (a2+b2) | (a3+b3) (a4+b4)
	 *
	 * 3rd iter(offset=1): ( (a1+b1) + (a3+b3)  ) | ( (a2+b2) +  (a4+b4) )
	 *
	 * 4th iter(offset=0): ( (a1+b1) + (a3+b3)  ) + ( (a2+b2) +  (a4+b4) )
	 *
	 * tx indicates the y-dimension in the matrix; ty indicates the x-dimension in the matrix
	 */
template<int BLOCK_SIZE, class T, int OP>
__global__
void reduce_to_col_kernel(const T* matrix, T* vector, int nCols, int nRows,
		T param, T factNew, T factOld) {

	__shared__ T shared[BLOCK_SIZE / 2][BLOCK_SIZE * 2];
	T sum;

	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int ty = threadIdx.y;
	int by = blockIdx.y;

	const int row_idx = by * gridDim.x * blockDim.x +   	// offset according to y index in grid
						bx * blockDim.x +  					// offset according to block index
						tx;									// offset in block

	if (row_idx >= nRows)
		return;
	int off = blockDim.y;

	if (OP==cuv::RF_MIN)
		sum = INT_MAX;
	else if(OP==cuv::RF_MAX)
		sum = -INT_MAX;
	else
		sum = 0.f;
//	printf("\n\nBlockDim.x: %d, BlockDim.y: %d\n", blockDim.x, blockDim.y);
//	printf("\n\ngridDim.x: %d, gridDim.y: %d\n", gridDim.x, gridDim.y);
//
//	printf("Thread.x: tx: %d, Thread.y: %d, Thread.z: %d\n", bx, tx, ty, threadIdx.z);
//	printf("Block.x: bx: %d, Block.y: %d\n", bx, by);
//	printf("nCols: %d, nRows: %d\n", nCols, nRows);
	for (int my = ty; my < nCols; my += off) {
//		if (my <100)
//			printf("my: %d\t", my);
		// to jump one col we proceed nRow elements in the matrix vector
		T f = matrix[my * nRows + row_idx ];

		if (OP == cuv::RF_ADD)
			sum += f;
		if (OP == cuv::RF_ADD_SQUARED)
			sum += f * f;
		if (OP == cuv::RF_MIN)
			sum = min(sum, f);
		if (OP == cuv::RF_MAX)
			sum = max(sum, f);
	}

	shared[ty][tx] = sum;

	__syncthreads();

	int offset = blockDim.y / 2;
	while (offset > 0) {
		if (ty < offset) {
			if (OP == cuv::RF_ADD)
				shared[ty][tx] += shared[ty + offset][tx];
			if (OP == cuv::RF_ADD_SQUARED)
				shared[ty][tx] += shared[ty + offset][tx];
			if (OP == cuv::RF_MIN)
				shared[ty][tx] = min(shared[ty][tx], shared[ty + offset][tx]);
			if (OP == cuv::RF_MAX)
				shared[ty][tx] = max(shared[ty][tx], shared[ty + offset][tx]);
		}
		offset >>= 1;
		__syncthreads();
	}
	
	if (ty == 0) {
		if (row_idx >= nCols){
			cuPrintf("Value is: %d\n", row_idx);
		}
		if (OP == cuv::RF_MIN || OP == cuv::RF_MAX)
			vector[row_idx] = shared[0][tx];
		else
			if(factOld != 0.f){
				if (isnan(vector[row_idx])){
					/*cuPrintf("Value of vector[%d]: %d\n", row_idx, vector[row_idx]);*/
				}
				if (isnan(shared[0][tx])){
					/*cuPrintf("Value of shared[0][%d]: %d\n", tx, shared[0][tx]);*/
				}
				vector[row_idx] = vector[row_idx] * factOld + shared[0][tx] * factNew;
			}else{
				if (isnan(shared[0][tx])){
					/*cuPrintf("Value of shared[0][%d]: %d\n", tx, shared[0][tx]);*/
				}
				vector[row_idx] = shared[0][tx] * factNew;
			}
		//vector[row_idx] = row_idx;
	}
	//__syncthreads();
}

template<int BLOCK_SIZE, class T, int OP>
__global__
void reduce_to_row_kernel(const T* matrix, T* vector, int nCols, int nRows,
		T param, T factNew, T factOld) {
	__shared__ T shared[BLOCK_SIZE * BLOCK_SIZE];
	const int tx = threadIdx.x, bx = blockIdx.x;
	const int ty = threadIdx.y, by = blockIdx.y;
	unsigned int idx = blockIdx.y * blockDim.y + threadIdx.y;
	for (; idx < nCols; idx += blockDim.y * gridDim.y) {
		int off = blockDim.x;

		if (OP == cuv::RF_MIN)
			shared[tx] = INT_MAX;
		else if (OP == cuv::RF_MAX)
			shared[tx] = -INT_MAX;
		else
			shared[tx] = 0;
		for (int my = tx; my < nRows; my += off) {
			T f = matrix[by * nRows + bx * blockDim.x + my];

			if (OP == cuv::RF_ADD)
				shared[tx] += f;
			if (OP == cuv::RF_ADD_SQUARED)
				shared[tx] += f * f;
			if (OP == cuv::RF_MAX)
				shared[tx] = max(shared[tx], f);
			if (OP == cuv::RF_MIN)
				shared[tx] = min(shared[tx], f);
		}
		__syncthreads();

		int offset = blockDim.x / 2;
		while (offset > 0) {
			if (tx < offset) {
				if (OP == cuv::RF_ADD)
					shared[tx] += shared[tx + offset];
				if (OP == cuv::RF_ADD_SQUARED)
					shared[tx] += shared[tx + offset];
				if (OP == cuv::RF_MAX)
					shared[tx] = max(shared[tx], shared[tx + offset]);
				if (OP == cuv::RF_MIN)
					shared[tx] = min(shared[tx], shared[tx + offset]);
			}
			offset >>= 1;
			__syncthreads();
		}

		if (tx == 0) {
			if (OP == cuv::RF_MIN || OP == cuv::RF_MAX)
				vector[by * blockDim.y + ty] = shared[0];
			else
				vector[by * blockDim.y + ty] = vector[by * blockDim.y + ty]
						* factOld + shared[0] * factNew;
		}
		__syncthreads();
	}
}

template<unsigned int BLOCK_DIM, class I, class T>
__global__
void argmax_row_kernel(I* vector, const T* matrix, unsigned int nCols, unsigned int nRows) {
	__shared__ I shIdx[BLOCK_DIM]; // index of the maximum
	__shared__ T shVal[BLOCK_DIM]; // value

	const unsigned int tx = threadIdx.x;
	const unsigned int by = blockIdx.x + gridDim.x*blockIdx.y;
	if (by >= nCols)
	   return;
	const unsigned int off = blockDim.x;

	unsigned int idx = by * nRows + tx;
	if (tx < nRows) {
	   shVal[tx] = (tx<nRows) ? matrix[idx] : (T) INT_MIN;
	   shIdx[tx] = (tx<nRows) ? tx          : 0;
	}

	for (unsigned int my = tx + off; my < nRows; my += off) {
	   idx += off;
	   T f = matrix[idx];

	   if (f > shVal[tx]) {
		  shVal[tx] = f;
		  shIdx[tx] = my;
	   }
	}
	__syncthreads();

	for (unsigned int offset = BLOCK_DIM/2 ; offset > 0; offset/=2) {
	   if (tx < offset) {
		   const unsigned int v = tx+offset;
		   if (shVal[tx] < shVal[v]) {
			   shVal[tx] = shVal[v];
			   shIdx[tx] = shIdx[v];
		   }
	   }
	}
	__syncthreads();

	if (tx == 0)
	   vector[by] = shIdx[0];
}

// "coalesced transpose" with no bank conflicts, example from SDK
// potential speedup by 5 possible for "fine-grained transpose"
template<int BLOCK_SIZE, class T>
__global__
void transpose_kernel(T* dst, const T* src, int width, int height) {
	const int bx = blockIdx.x * blockDim.x;
	const int by = blockIdx.y * blockDim.y;
	const int tx = bx + threadIdx.x;
	const int ty = by + threadIdx.y;

	__shared__
	T shared[BLOCK_SIZE][BLOCK_SIZE + 1];

	if (tx < width && ty < height) {
		shared[threadIdx.y][threadIdx.x] = src[ty * width + tx];
	}
	__syncthreads();

	if (by + threadIdx.x < height && threadIdx.y + bx < width) {
		dst[(bx + threadIdx.y) * height + by + threadIdx.x]
				= shared[threadIdx.x][threadIdx.y];
	}
}

namespace cuv {
template<class __value_type, class __memory_space_type, class __index_type>
dense_matrix<__value_type , column_major, __memory_space_type, __index_type >*blockview(
		dense_matrix<__value_type,column_major,__memory_space_type,__index_type>& matrix,
				__index_type start_rows,
				__index_type num_rows,
				__index_type start_cols,
				__index_type num_cols,
				column_major
				) {
			cuvAssert(start_rows==0);
			cuvAssert(num_rows==matrix.h())
			return new dense_matrix<__value_type,column_major,__memory_space_type,__index_type>(num_rows,num_cols, matrix.ptr()+matrix.h()*start_cols,true);
		}

template<class __value_type, class __memory_space_type, class __index_type>
dense_matrix<__value_type,row_major,__memory_space_type,__index_type>* blockview(
		dense_matrix<__value_type,row_major,__memory_space_type,__index_type>& matrix,
		__index_type start_rows,
		__index_type num_rows,
		__index_type start_cols,
		__index_type num_cols,
		row_major
) {
	cuvAssert(start_cols==0);
	cuvAssert(num_cols==matrix.w())
	return new dense_matrix<__value_type,row_major,__memory_space_type,__index_type>(num_rows,num_cols, matrix.ptr()+matrix.w()*start_rows,true);
}
template<class __value_type, class __memory_layout, class __memory_space_type, class __index_type>
dense_matrix<__value_type,__memory_layout,__memory_space_type,__index_type>* blockview(
		dense_matrix<__value_type,__memory_layout,__memory_space_type,__index_type> & matrix,
		__index_type start_rows,
		__index_type num_rows ,
		__index_type start_cols,
		__index_type num_cols) {
	return blockview(matrix,start_rows,num_rows,start_cols,num_cols, __memory_layout());
}

__global__ void bitflip_kernel(float* M, int height, int row, int n) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int off = blockDim.x * gridDim.x;
    for (unsigned int i = idx; i < n; i += off){
		M[i * height + row] = 1 - M[i * height + row];
	}

}

namespace bitflip_row_impl{
	template<class V, class I>
	void bitflip(dense_matrix<V,column_major,dev_memory_space,I>& m, const I& row){
		int num_threads = (int) min(512.f, ceil((float)sqrt(m.w())));
		int num_blocks  = (int) ceil((float)m.w()/num_threads);
		bitflip_kernel<<<num_blocks,num_threads>>>(m.ptr(),m.h(),row, m.w());
		cuvSafeCall(cudaThreadSynchronize());
	}
	template<class V, class I>
	void bitflip(dense_matrix<V,column_major,host_memory_space,I>& m, const I& row){
		for(int i=0;i<m.w();i++)
			m.set(row,i,(V)(1.f-m(row,i)));
	}
}
// bitflip a row of a column-major matrix
template<class __value_type, class __memory_layout, class __memory_space_type, class __index_type>
void bitflip(dense_matrix<__value_type,__memory_layout,__memory_space_type,__index_type>& matrix,
		__index_type row){
		assert(row<matrix.h());
		assert(matrix.ptr());
		bitflip_row_impl::bitflip(matrix,row);
}

/// column major blas3
template<>
void prod(dense_matrix<float,column_major,dev_memory_space>& dst,
		dense_matrix<float,column_major,dev_memory_space>& A,
		dense_matrix<float,column_major,dev_memory_space>& B,
		char transA,
		char transB,
		const float& factAB,
		const float& factC) {
	int m = (transA=='t' ? A.w() : A.h());
	int k1 = (transA=='t' ? A.h() : A.w());
	int k2 = (transB=='t' ? B.w() : B.h());
	int n = (transB=='t' ? B.h() : B.w());

	cuvAssert(dst.h() == m);
	cuvAssert(dst.w() == n);
	cuvAssert(k1 == k2);
	cuvAssert(A.ptr());
	cuvAssert(B.ptr());
	cuvAssert(dst.ptr());

	cublasSgemm(transA, transB, m, n, k1, factAB, A.ptr(), A.h(),B.ptr(), B.h(), factC, dst.ptr(), dst.h());
	cuvAssert( cublasGetError() == CUBLAS_STATUS_SUCCESS );
	cuvSafeCall(cudaThreadSynchronize());
}

template<>
void prod(dense_matrix<float,column_major,host_memory_space>& dst,
		dense_matrix<float,column_major,host_memory_space>& A,
		dense_matrix<float,column_major,host_memory_space>& B,
		char transA,
		char transB,
		const float& factAB,
		const float& factC) {
	int m = (transA=='t' ? A.w() : A.h());
	int k1 = (transA=='t' ? A.h() : A.w());
	int k2 = (transB=='t' ? B.w() : B.h());
	int n = (transB=='t' ? B.h() : B.w());

	cuvAssert(dst.h() == m);
	cuvAssert(dst.w() == n);
	cuvAssert(k1 == k2);
	cuvAssert(A.ptr() != NULL);
	cuvAssert(B.ptr() != NULL);
	cuvAssert(dst.ptr());

#if 1 /* CBLAS */
	cblas_sgemm(
			CblasColMajor,
			CVT_TRANSPOSE(transA),
			CVT_TRANSPOSE(transB), m, n, k1,
			factAB, A.ptr(), A.h(),B.ptr(), B.h(), factC, dst.ptr(), dst.h());
#else /* naive */
	for(int i=0; i<A.h();i++)
	for(int j=0; j<B.w(); j++) {
		float f=0;
		for(int k=0;k<A.w();k++) {
			f += A(i,k)*B(k,j);
		}
		dst.set(i,j,f);
	}
#endif
}
/// row major blas3
template<>
void prod(dense_matrix<float,row_major,dev_memory_space>& dst,
		dense_matrix<float,row_major,dev_memory_space>& A,
		dense_matrix<float,row_major,dev_memory_space>& B,
		char transA,
		char transB,
		const float& factAB,
		const float& factC) {
	// we use column major prod and just exchange width and height
	int m = (transB=='t' ? B.h() : B.w());
	int k1 = (transB=='t' ? B.w() : B.h());
	int k2 = (transA=='t' ? A.h() : A.w());
	int n = (transA=='t' ? A.w() : A.h());

	cuvAssert(dst.h() == n);
	cuvAssert(dst.w() == m);
	cuvAssert(k1 == k2);
	cuvAssert(A.ptr());
	cuvAssert(B.ptr());
	cuvAssert(dst.ptr());
	cublasSgemm(transB, transA, m, n, k1, factAB, B.ptr(), B.w(),A.ptr(), A.w(), factC, dst.ptr(), dst.w());

	cuvAssert( cublasGetError() == CUBLAS_STATUS_SUCCESS );
	cuvSafeCall(cudaThreadSynchronize());
}

template<>
void prod(dense_matrix<float,row_major,host_memory_space>& dst,
		dense_matrix<float,row_major,host_memory_space>& A,
		dense_matrix<float,row_major,host_memory_space>& B,
		char transA,
		char transB,
		const float& factAB,
		const float& factC) {
	int m = (transA=='t' ? A.w() : A.h());
	int k1 = (transA=='t' ? A.h() : A.w());
	int k2 = (transB=='t' ? B.w() : B.h());
	int n = (transB=='t' ? B.h() : B.w());

	cuvAssert(dst.h() == m);
	cuvAssert(dst.w() == n);
	cuvAssert(k1 == k2);
	cuvAssert(A.ptr() != NULL);
	cuvAssert(B.ptr() != NULL);
	cuvAssert(dst.ptr());

	cblas_sgemm(
			CblasRowMajor,
			CVT_TRANSPOSE(transA),
			CVT_TRANSPOSE(transB), m, n, k1,
			factAB, A.ptr(), A.w(),B.ptr(), B.w(), factC, dst.ptr(), dst.w());
}

template<class V, class I, class V2, class OP>
__global__
void matrix_plus_vector_kernel_column_major(V*A,V2* v,I w,I h, OP op) {
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if(tid>h) return;
	V2 tid_v = v[tid];
	for(int i=tid;i<w;i++)
	A[i] = op(A[i],tid_v);
}
template<class V, class I, class V2, class OP>
__global__
void matrix_plus_vector_kernel_column_major2 (V *A, const V2* v, I h, I w, OP op) {
	const unsigned int idx = __mul24(blockIdx.x , blockDim.x) + threadIdx.x;
	const unsigned int numThreads = __mul24(blockDim.x , gridDim.x);

	int stop = w*h;
	for (unsigned int i = idx; i < stop; i += numThreads)
	A[i] = op(A[i] , v[i % h]);
}
template<class V, class I, class V2, class OP>
__global__
void matrix_plus_vector_kernel_row_major (V *A, V2* v, I h, I w, OP op) {
	__shared__ V scalar;
	for(unsigned int baseidx = blockIdx.x; baseidx < h; baseidx += gridDim.x) {
		if (threadIdx.x == 0) {
			scalar = (V) v[baseidx];
		}
		__syncthreads();
		for (unsigned int i = threadIdx.x; i < w; i += blockDim.x) {
			const unsigned int k = baseidx * w + i;
			A[k] = op(A[k] , scalar);
		}
		__syncthreads(); // necessary, otherwise the threads use different values of scalar!
	}
}

namespace matrix_plus_vector_impl {
	template<class V, class I, class V2, class OP>
	void matrix_plus_col(dense_matrix<V,row_major,dev_memory_space,I>& A, const vector<V2,dev_memory_space,I>& v, const OP& op) {
		cuvAssert(A.h() == v.size());
		const unsigned int num_threads = min(512,A.w());
		const unsigned int num_blocks  = min(1024,A.h());
		matrix_plus_vector_kernel_row_major<<<num_blocks,num_threads>>>(A.ptr(), v.ptr(), A.h(), A.w(), op);
		cuvSafeCall(cudaThreadSynchronize());
	}
	template<class V, class I, class V2, class OP>
	void matrix_plus_col(dense_matrix<V,column_major,dev_memory_space,I>& A, const vector<V2,dev_memory_space,I>& v, const OP& op) {
		cuvAssert(A.h() == v.size());
		const unsigned int num_threads = 512;
		const unsigned int num_blocks  = min(512,(int)ceil((float)A.n() / num_threads));
		matrix_plus_vector_kernel_column_major2<<<num_blocks,num_threads>>>(A.ptr(), v.ptr(), A.h(), A.w(), op);
		cuvSafeCall(cudaThreadSynchronize());
	}
	template<class V, class I, class V2, class OP>
	void matrix_plus_col(dense_matrix<V,column_major,host_memory_space,I>& A, const vector<V2,host_memory_space,I>& v, const OP& op) {
		const V2* v_ptr = v.ptr();
		V * A_ptr = A.ptr();
		for(int j=0;j<A.w();j++) {
			v_ptr = v.ptr();
			for(int i=0;i<A.h();i++,A_ptr++,v_ptr++)
			*A_ptr = op(*A_ptr,*v_ptr);
		}
	}
	template<class V, class I, class V2, class OP>
	void matrix_plus_col(dense_matrix<V,row_major,host_memory_space,I>& A, const vector<V2,host_memory_space,I>& v, const OP& op) {
		const V2* v_ptr = v.ptr();
		V * A_ptr = A.ptr();
		for(int i=0;i<A.h();i++, v_ptr++) {
			for(int j=0;j<A.w();j++)
			*A_ptr++ = op(*A_ptr,*v_ptr);
		}
	}
}

template<class __matrix_type, class __vector_type>
void matrix_plus_col(__matrix_type& A, const __vector_type& v) {
	matrix_plus_vector_impl::matrix_plus_col(A,v, thrust::plus<typename __matrix_type::value_type>());
}
template<class __matrix_type, class __vector_type>
void matrix_times_col(__matrix_type& A, const __vector_type& v) {
	matrix_plus_vector_impl::matrix_plus_col(A,v, thrust::multiplies<typename __matrix_type::value_type>());
}
template<class __matrix_type, class __vector_type>
void matrix_divide_col(__matrix_type& A, const __vector_type& v) {
	matrix_plus_vector_impl::matrix_plus_col(A,v, thrust::divides<typename __matrix_type::value_type>());
}

namespace reduce_to_col_impl {

	template<int rf,class V,class I, class V2>
	void reduce_to_col(vector<V2,host_memory_space,I>&v, const dense_matrix<V,column_major,host_memory_space,I>& m, const V& factNew, const V& factOld) {
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.h() == v.size());
		const V* A_ptr = m.ptr();

		V2* v_ptr = v.ptr();
		for(int j=0; j<m.h(); j++) {
			if(0);
			else if(rf == RF_MIN) *v_ptr++ = INT_MAX;
			else if(rf == RF_MAX) *v_ptr++ = -INT_MAX;
			else *v_ptr++ *= factOld;
		}
		for(int i=0;i<m.w();i++) {
			v_ptr = v.ptr();
			for(int j=0; j<m.h(); j++,v_ptr++,A_ptr++) {
				if(0);
				else if(rf==RF_ADD) *v_ptr += factNew * *A_ptr;
				else if(rf==RF_ADD_SQUARED) *v_ptr += factNew * (*A_ptr * *A_ptr);
				else if(rf==RF_MIN) *v_ptr = min(*A_ptr, *v_ptr);
				else if(rf==RF_MAX) *v_ptr = max(*A_ptr, *v_ptr);
			}
		}
	}
	template<int rf,class V,class I, class V2>
	void reduce_to_col(vector<V2,host_memory_space,I>&v, const dense_matrix<V,row_major,host_memory_space,I>& m, const V& factNew, const V& factOld) {
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.h() == v.size());
		const V* A_ptr = m.ptr();

		V2* v_ptr = v.ptr();
		for(int j=0; j<m.h(); j++) {
			if(0);
			else if(rf == RF_MIN) *v_ptr++ = INT_MAX;
			else if(rf == RF_MAX) *v_ptr++ = -INT_MAX;
			else *v_ptr++ *= factOld;
		}
		v_ptr = v.ptr();
		for(int i=0;i<m.h(); i++) {
			for(int j=0; j<m.w(); j++,A_ptr++) {
				if(0);
				else if(rf==RF_ADD) *v_ptr += factNew * *A_ptr;
				else if(rf==RF_ADD_SQUARED) *v_ptr += factNew * (*A_ptr * *A_ptr);
				else if(rf==RF_MIN) *v_ptr = min(*A_ptr, *v_ptr);
				else if(rf==RF_MAX) *v_ptr = max(*A_ptr, *v_ptr);
			}
			v_ptr++;
		}
	}
	template<int rf,class V,class I, class V2>
	void reduce_to_col(vector<V2,dev_memory_space,I>&v, const dense_matrix<V,column_major,dev_memory_space,I>& m, const V& factNew, const V& factOld) {
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.h() == v.size());
		static const int BLOCK_SIZE = 16;
		static const int BLOCK_DIM_X = BLOCK_SIZE*2;
		static const int BLOCK_DIM_Y = BLOCK_SIZE/2;
		const int blocks_needed = ceil((float)m.h()/(BLOCK_DIM_X));
		int grid_x =0, grid_y=0;

		// how to handle grid dimension constraint
		if (blocks_needed <= 65535){
			grid_x = blocks_needed;
			grid_y = 1;
		}else{
			// try to avoid large noop blocks by adjusting x and y dimension to nearly equal size
			grid_x = ceil(sqrt(blocks_needed));
			grid_y = ceil((float)blocks_needed/grid_x);
		}
		dim3 grid(grid_x, grid_y);
		dim3 threads(BLOCK_DIM_X,BLOCK_DIM_Y);
		cudaPrintfInit();		
		reduce_to_col_kernel<BLOCK_SIZE,V,rf><<<grid,threads>>>(m.ptr(),v.ptr(),m.w(),m.h(),0,factNew,factOld);
		cuvSafeCall(cudaThreadSynchronize());
		cudaPrintfDisplay(NULL, true);
		cudaPrintfEnd();
	}
	template<int rf,class V,class I, class V2>
	void reduce_to_col(vector<V2,dev_memory_space,I>&v, const dense_matrix<V,row_major,dev_memory_space,I>& m, const V& factNew, const V& factOld) {
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.h() == v.size());
		/*static const int BLOCK_SIZE = 16;*/
		/*dim3 grid(1, m.h());                           //TODO: make reduce_to_row kernel use grids*/
		/*dim3 threads(BLOCK_SIZE*BLOCK_SIZE,1); // not working for m.h() > 65535*/
		/*// yes, we abuse the reduce_to_row kernel here :)*/
		/*reduce_to_row_kernel<BLOCK_SIZE,V,rf><<<grid,threads>>>(m.ptr(),v.ptr(),m.h(),m.w(),0,factNew,factOld);*/
		cuvAssert(rf == RF_ADD); // what else does alex support?

		vector<V2,dev_memory_space> w(v.size());

		if(factOld != 0.0f)
			copy(w, v);

		NVMatrix tmp(const_cast<V*>(m.ptr()),(int)m.h(),(int)m.w(),false);
		NVMatrix dst(v.ptr(),(int)v.size(),(int)1,false);
		tmp.aggregate(1,dst,256,NVMatrix::SUM);

		if(factNew != 1.0f)
			apply_scalar_functor(v, SF_MULT, factNew);

		if(factOld != 0.0f)
			apply_binary_functor(v, w, BF_XPBY, factOld);

		cuvSafeCall(cudaThreadSynchronize());
	}

}
template<class __matrix_type, class __vector_type>
void reduce_to_col(__vector_type&v, const __matrix_type& m, reduce_functor rf, const typename __matrix_type::value_type& factNew, const typename __matrix_type::value_type& factOld) {
	switch(rf) {
		case RF_ADD:
		reduce_to_col_impl::reduce_to_col<RF_ADD>(v,m,factNew,factOld);
		break;
		case RF_ADD_SQUARED:
		reduce_to_col_impl::reduce_to_col<RF_ADD_SQUARED>(v,m,factNew,factOld);
		break;
		case RF_MIN:
		reduce_to_col_impl::reduce_to_col<RF_MIN>(v,m,factNew,factOld);
		break;
		case RF_MAX:
		reduce_to_col_impl::reduce_to_col<RF_MAX>(v,m,factNew,factOld);
		break;
		default:
		throw std::runtime_error("supplied reduce_functor does not exist");
	}
}

namespace reduce_to_row_impl {
	template<int rf,class V,class I, class V2>
	void reduce_to_row(vector<V2,host_memory_space,I>&v, const dense_matrix<V,row_major,host_memory_space,I>& m, const V& factNew, const V& factOld) {
		cuvAssert(rf==RF_ADD);
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.w() == v.size());
		const V* A_ptr = m.ptr();

		V2* v_ptr = v.ptr();
		for(int j=0; j<v.size(); j++) {
			*v_ptr++ *= factOld;
		}
		for(int i=0;i<m.h();i++) {
			v_ptr = v.ptr();
			for(int j=0; j<m.w(); j++) {
				*v_ptr++ += factNew * *A_ptr++;
			}
		}
	}
	template<int rf,class V,class I, class V2>
	void reduce_to_row(vector<V2,host_memory_space,I>&v, const dense_matrix<V,column_major,host_memory_space,I>& m, const V& factNew, const V& factOld) {
		cuvAssert(rf==RF_ADD);
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.w() == v.size());
		const V* A_ptr = m.ptr();

		V2* v_ptr = v.ptr();
		for(int j=0; j<v.size(); j++) {
			*v_ptr++ *= factOld;
		}
		v_ptr = v.ptr();
		for(int i=0;i<m.w();i++) {
			for(int j=0; j<m.h(); j++)
			*v_ptr += factNew * *A_ptr++;
			v_ptr++;
		}
	}

	template<int rf, class V,class I, class V2>
	void reduce_to_row(vector<V2,dev_memory_space,I>&v, const dense_matrix<V,column_major,dev_memory_space,I>& m, const V& factNew, const V& factOld) {
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.w() == v.size());
		static const int BLOCK_SIZE = 16;
		dim3 grid(1, m.w());
		dim3 threads(BLOCK_SIZE*BLOCK_SIZE,1);
		reduce_to_row_kernel<BLOCK_SIZE,V,rf><<<grid,threads>>>(m.ptr(),v.ptr(),m.w(),m.h(),0,factNew,factOld);
		cuvSafeCall(cudaThreadSynchronize());
	}

	template<int rf,class V,class I, class V2>
	void reduce_to_row(vector<V2,dev_memory_space,I>&v, const dense_matrix<V,row_major,dev_memory_space,I>& m, const V& factNew, const V& factOld) {
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.w() == v.size());
		static const long int BLOCK_SIZE = 16;
		const int blocks_needed = ceil((float)m.w()/(BLOCK_SIZE*2));

		int grid_x =0, grid_y=0;

		// how to handle grid dimension constraint
		if (blocks_needed <= 65535){
			grid_x = blocks_needed;
			grid_y = 1;
		}else{
			// try to avoid large noop blocks by adjusting x and y dimension to nearly equal size
			grid_x = ceil(sqrt(blocks_needed));
			grid_y = ceil((float)blocks_needed/grid_x);
		}

		dim3 grid(grid_x, grid_y);
		dim3 threads(BLOCK_SIZE*2, BLOCK_SIZE/2);
		// yes, we abuse the reduce_to_col kernel here :)
		reduce_to_col_kernel<BLOCK_SIZE,V,rf><<<grid,threads>>>(m.ptr(),v.ptr(),m.h(),m.w(),0,factNew,factOld);
		cuvSafeCall(cudaThreadSynchronize());
	}

}
template<class __matrix_type, class __vector_type>
void reduce_to_row(__vector_type&v, const __matrix_type& m,reduce_functor rf, const typename __matrix_type::value_type& factNew, const typename __matrix_type::value_type& factOld) {
	switch(rf) {
		case RF_ADD:
		reduce_to_row_impl::reduce_to_row<RF_ADD>(v,m,factNew,factOld);
		break;
		case RF_ADD_SQUARED:
		reduce_to_row_impl::reduce_to_row<RF_ADD_SQUARED>(v,m,factNew,factOld);
		break;
		case RF_MIN:
		reduce_to_row_impl::reduce_to_row<RF_MIN>(v,m,factNew,factOld);
		break;
		case RF_MAX:
		reduce_to_row_impl::reduce_to_row<RF_MAX>(v,m,factNew,factOld);
		break;
		default:
		throw std::runtime_error("supplied reduce_functor does not exist");
	}
}

namespace argmax_to_XXX_impl{

template<class V, class V2, class I>
void argmax_to_row(vector<V2,dev_memory_space>&v, const dense_matrix<V,column_major, dev_memory_space, I>& m) {
	cuvAssert(m.ptr() != NULL);
	cuvAssert(m.w() == v.size());
	const unsigned int u = min(m.w(), MAX_GRID_SIZE);
	dim3 grid(u, ceil(m.w()/(float)u));
	static const unsigned int BLOCK_DIM = 256;
	argmax_row_kernel<BLOCK_DIM><<<grid,BLOCK_DIM>>>(v.ptr(),m.ptr(),m.w(),m.h());
	cuvSafeCall(cudaThreadSynchronize());
}

template<class V, class V2, class I>
void argmax_to_column(vector<V2,dev_memory_space, I>&v, const dense_matrix<V,row_major,dev_memory_space,I>& m) {
	cuvAssert(m.ptr() != NULL);
	cuvAssert(m.h() == v.size());
	const unsigned int u = min(m.h(), MAX_GRID_SIZE);
	dim3 grid(u, ceil(m.h()/(float)u));
	static const unsigned int BLOCK_DIM = 256;
	argmax_row_kernel<BLOCK_DIM><<<grid,BLOCK_DIM>>>(v.ptr(),m.ptr(),m.h(),m.w());
	cuvSafeCall(cudaThreadSynchronize());
}

template<class V, class V2, class I>
void argmax_to_row(vector<V2,host_memory_space,I>&v, const dense_matrix<V,column_major, host_memory_space,I>& m) {
	cuvAssert(m.ptr() != NULL);
	cuvAssert(m.w() == v.size());
	const V* ptr = m.ptr();
	V2* res = v.ptr();
	for(int i=0; i<m.w();i++) {
		int idx = 0;
		V val = *ptr;
		for(int j=0; j<m.h();j++) {
			if(*ptr > val) {
				val = *ptr;
				idx = j;
			}
			ptr++;
		}
		*res++ = idx;
	}
}

template<class V, class V2, class I>
void argmax_to_column(vector<V2,host_memory_space,I>&v, const dense_matrix<V,row_major,host_memory_space,I>& m) {
	cuvAssert(m.ptr() != NULL);
	cuvAssert(m.h() == v.size());
	const V* ptr = m.ptr();
	V2* res = v.ptr();
	for(int i=0; i<m.h();i++) {
		int idx = 0;
		V val = *ptr;
		for(int j=0; j<m.w();j++) {
			if(*ptr > val) {
				val = *ptr;
				idx = j;
			}
			ptr++;
		}
		*res++ = idx;
	}
}

}

template<class V, class M>
void argmax_to_column(V&v, const M& m) {
	argmax_to_XXX_impl::argmax_to_column(v,m);
}
template<class V, class M>
void argmax_to_row(V&v, const M& m) {
	argmax_to_XXX_impl::argmax_to_row(v,m);
}

namespace transpose_impl{

template<class V, class I>
void transpose(dense_matrix<V,column_major, dev_memory_space, I>& dst,
		 const dense_matrix<V,column_major, dev_memory_space, I>& src) {
	cuvAssert(dst.w() == src.h());
	cuvAssert(dst.h() == src.w());
	const I width = dst.w();
	const I height = dst.h();
	static const int BLOCK_SIZE = 16;
	const int numBlocksX = ceil((float)width / BLOCK_SIZE);
	const int numBlocksY = ceil((float)height / BLOCK_SIZE);
	dim3 gridSize(numBlocksX, numBlocksY, 1);
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
	transpose_kernel<BLOCK_SIZE><<<gridSize, blockSize>>>(dst.ptr(), src.ptr(), width, height);
	cuvSafeCall(cudaThreadSynchronize());
}

template<class V, class I>
void transpose(dense_matrix<V,row_major,dev_memory_space, I>& dst,
		 const dense_matrix<V,row_major,dev_memory_space, I>& src) {
	cuvAssert(dst.w() == src.h());
	cuvAssert(dst.h() == src.w());
	const I width = dst.h();
	const I height = dst.w();
	static const int BLOCK_SIZE = 16;
	const int numBlocksX = ceil((float)width / BLOCK_SIZE);
	const int numBlocksY = ceil((float)height / BLOCK_SIZE);
	dim3 gridSize(numBlocksX, numBlocksY, 1);
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
	transpose_kernel<BLOCK_SIZE><<<gridSize, blockSize>>>(dst.ptr(), src.ptr(), width, height);
	cuvSafeCall(cudaThreadSynchronize());
}

template<class V, class I>
void transpose(dense_matrix<V,column_major,host_memory_space, I>& dst,
		 const dense_matrix<V,column_major,host_memory_space, I>& src) {
	cuvAssert(dst.w() == src.h());
	cuvAssert(dst.h() == src.w());
	V* dst_ptr = dst.ptr();
	const V* src_ptr = src.ptr();
	for(int i=0; i<dst.w(); i++) {
		for(int j=0; j<dst.h(); j++) {
			*dst_ptr++ = src_ptr[j*src.h()];
		}
		src_ptr++;
	}
}

template<class V, class I>
void transpose(dense_matrix<V,row_major,host_memory_space, I>& dst,
		 const dense_matrix<V,row_major,host_memory_space, I>& src) {
	cuvAssert(dst.w() == src.h());
	cuvAssert(dst.h() == src.w());
	V* dst_ptr = dst.ptr();
	const V* src_ptr = src.ptr();
	for(int i=0; i<dst.h(); i++) {
		for(int j=0; j<dst.w(); j++) {
			*dst_ptr++ = src_ptr[j*src.w()];
		}
		src_ptr++;
	}
}
}

template<class M>
void transpose(M& dst, const M& src){
	transpose_impl::transpose(dst,src);
}

#define INSTANTIATE_MV(V,M) \
  template void matrix_plus_col(dense_matrix<V,M,dev_memory_space>&, const vector<V,dev_memory_space>&);   \
  template void matrix_plus_col(dense_matrix<V,M,host_memory_space>&, const vector<V,host_memory_space>&); \
  template void matrix_times_col(dense_matrix<V,M,dev_memory_space>&, const vector<V,dev_memory_space>&);  \
  template void matrix_times_col(dense_matrix<V,M,host_memory_space>&, const vector<V,host_memory_space>&); \
  template void matrix_divide_col(dense_matrix<V,M,dev_memory_space>&, const vector<V,dev_memory_space>&);  \
  template void matrix_divide_col(dense_matrix<V,M,host_memory_space>&, const vector<V,host_memory_space>&);

#define INSTANTIATE_REDCOL(V,M) \
  template void reduce_to_row(vector<V,dev_memory_space>&, const dense_matrix<V,M,dev_memory_space>&, reduce_functor,  const V&,const V&); \
  template void reduce_to_col(vector<V,dev_memory_space>&, const dense_matrix<V,M,dev_memory_space>&, reduce_functor, const V&,const V&); \
  template void reduce_to_row(vector<V,host_memory_space>&, const dense_matrix<V,M,host_memory_space>&, reduce_functor,  const V&,const V&); \
  template void reduce_to_col(vector<V,host_memory_space>&, const dense_matrix<V,M,host_memory_space>&,reduce_functor,  const V&,const V&);

#define INSTANTIATE_REDROW(V,M) \
  template void reduce_to_col(vector<V,dev_memory_space>&, const dense_matrix<V,M,dev_memory_space>&, reduce_functor, const V&,const V&); \
  template void reduce_to_row(vector<V,dev_memory_space>&, const dense_matrix<V,M,dev_memory_space>&, reduce_functor,  const V&,const V&); \
  template void reduce_to_col(vector<V,host_memory_space>&, const dense_matrix<V,M,host_memory_space>&, reduce_functor, const V&,const V&); \
  template void reduce_to_row(vector<V,host_memory_space>&, const dense_matrix<V,M,host_memory_space>&, reduce_functor,  const V&,const V&);

#define INSTANTIATE_BLOCKVIEW(V,M,I) \
  template dense_matrix<V,M,host_memory_space,I>* blockview(dense_matrix<V,M,host_memory_space,I>&,I,I,I,I); \
  template dense_matrix<V,M, dev_memory_space,I>* blockview(dense_matrix<V,M, dev_memory_space,I>&,I,I,I,I);

#define INSTANTIATE_TRANSPOSE(V,M,I) \
  template void transpose(dense_matrix<V,M,host_memory_space,I>&,const dense_matrix<V,M,host_memory_space,I>&); \
  template void transpose(dense_matrix<V,M,dev_memory_space,I>&,const dense_matrix<V,M,dev_memory_space,I>&); 

#define INSTANTIATE_ARGMAX_TO_ROW(V,M,I) \
  template void argmax_to_row(vector<int,dev_memory_space,I>&,const dense_matrix<V,M,dev_memory_space,I>&);   \
  template void argmax_to_row(vector<int,host_memory_space,I>&,const dense_matrix<V,M,host_memory_space,I>&);  \
  template void argmax_to_row(vector<float,dev_memory_space,I>&,const dense_matrix<V,M,dev_memory_space,I>&);   \
  template void argmax_to_row(vector<float,host_memory_space,I>&,const dense_matrix<V,M,host_memory_space,I>&);  
#define INSTANTIATE_ARGMAX_TO_COL(V,M,I) \
  template void argmax_to_column(vector<int,dev_memory_space,I>&,const dense_matrix<V,M,dev_memory_space,I>&);   \
  template void argmax_to_column(vector<int,host_memory_space,I>&,const dense_matrix<V,M,host_memory_space,I>&); \
  template void argmax_to_column(vector<float,dev_memory_space,I>&,const dense_matrix<V,M,dev_memory_space,I>&);   \
  template void argmax_to_column(vector<float,host_memory_space,I>&,const dense_matrix<V,M,host_memory_space,I>&);   


INSTANTIATE_ARGMAX_TO_COL(float,row_major,unsigned int);
INSTANTIATE_ARGMAX_TO_COL(int,row_major,unsigned int);

INSTANTIATE_ARGMAX_TO_ROW(float,column_major,unsigned int);
INSTANTIATE_ARGMAX_TO_ROW(int,column_major,unsigned int);

INSTANTIATE_TRANSPOSE(float,column_major,unsigned int);
INSTANTIATE_TRANSPOSE(float,row_major,unsigned int);
INSTANTIATE_TRANSPOSE(int,column_major,unsigned int);
INSTANTIATE_TRANSPOSE(int,row_major,unsigned int);

INSTANTIATE_MV(float, column_major);
INSTANTIATE_MV(float, row_major);

INSTANTIATE_REDCOL(float,column_major);
INSTANTIATE_REDROW(float,row_major);

INSTANTIATE_BLOCKVIEW(float,column_major,unsigned int);
INSTANTIATE_BLOCKVIEW(float,row_major,unsigned int);

template void bitflip(dense_matrix<float,column_major,host_memory_space>&, unsigned int);
template void bitflip(dense_matrix<float,column_major,dev_memory_space>&, unsigned int);

}; // cuv
