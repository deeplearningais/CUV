#include <iostream>
#include <host_dia_matrix.hpp>
#include <dev_dia_matrix.hpp>
#include "matrix_ops.hpp"
#include <texture.h>

using namespace std;

// stuff from NVIDIA SDK
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define small_grid_thread_id(void) ((__umul24(blockDim.x, blockIdx.x) + threadIdx.x))
#define large_grid_thread_id(void) ((__umul24(blockDim.x,blockIdx.x + __umul24(blockIdx.y,gridDim.x)) + threadIdx.x))
#define large_grid_thread_num(void) ((__umul24(blockDim.x,gridDim.x + __umul24(blockDim.y,gridDim.y))))


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
		/****************************************************************
		 *   Device Code
		 ****************************************************************/
		template <typename value_type, typename index_type, unsigned int BLOCK_SIZE, bool UseCache>
			__global__ void
			spmv_dia_kernel_trans(const index_type num_rows, 
					const index_type num_cols, 
					const index_type num_diagonals,
					const index_type stride,
					const int        * diagonal_offsets,
					const value_type * values,
					const value_type * v, 
					value_type       * dst)
			{
				__shared__ int        offsets[BLOCK_SIZE];

				const index_type thread_id = large_grid_thread_id();
				const index_type grid_size = large_grid_thread_num();

				// load diagonal offsets into shared memory
				if(threadIdx.x < num_diagonals)
					offsets[threadIdx.x] = diagonal_offsets[threadIdx.x];

				__syncthreads();

				for(index_type col = thread_id; col < num_cols; col += grid_size)
				{
					value_type sum = dst[col];
					index_type offset = 0;

					for(index_type n = 0; n < num_diagonals; n++, offset+=stride)
					{
						const int row = col - offsets[n];

						if(row >= 0 && row < num_rows)
						{
							const value_type A_ij = values[       offset + row];
							sum += A_ij * fetch_x<UseCache>(row, v);
						}
					}
					dst[col] = sum;
				}
			}
		template <typename value_type, typename index_type, unsigned int BLOCK_SIZE, bool UseCache>
			__global__ void
			spmv_dia_kernel2(const index_type num_rows, 
					const index_type num_cols, 
					const index_type num_diagonals,
					const index_type stride,
					const int        * diagonal_offsets,
					const value_type * values,
					const value_type * v, 
					value_type * dst)
			{
				__shared__ int offsets[BLOCK_SIZE];

				const index_type thread_id = large_grid_thread_id();
				const index_type grid_size = large_grid_thread_num();

				// load diagonal offsets into shared memory
				if(threadIdx.x < num_diagonals)
					offsets[threadIdx.x] = diagonal_offsets[threadIdx.x];

				__syncthreads();

				for(index_type row = thread_id; row < num_rows; row += grid_size)
				{
					value_type sum = dst[row];
					index_type offset = row;
					for(index_type n = 0; n < num_diagonals; n++, offset+=stride)
					{
						const int col = row + offsets[n];
						if(col >= 0 && col < num_cols)
						{
							const value_type A_ij = values[       offset];
							sum += A_ij * fetch_x<UseCache>(col, v);
						}
					}
					dst[row] = sum;
				}
			}

		template <typename value_type, typename index_type>
			void spmv_dia_device(const dev_dia_matrix<value_type,index_type>& A, 
					const dev_vector<value_type>& v, 
					dev_vector<value_type>& dst, 
					char transA,
					const value_type& factAv,
					const value_type& factC)
			{

				const unsigned int BLOCK_SIZE = 256;
				if(transA != 't'){
					const dim3 grid = make_large_grid(A.h(),BLOCK_SIZE);
					cuvAssert(A.num_dia() < BLOCK_SIZE); // kernel doesn't handle larger numbers of diagonals
					spmv_dia_kernel2<value_type, index_type, BLOCK_SIZE, false> <<<grid, BLOCK_SIZE>>>
						(A.h(), A.w(),  A.num_dia(),  A.stride(),
						 A.get_offsets().ptr(), A.vec()->ptr(),
						 v.ptr(), dst.ptr());
				}else{
					const dim3 grid = make_large_grid(A.w(),BLOCK_SIZE);
					cuvAssert(A.num_dia() < BLOCK_SIZE); // kernel doesn't handle larger numbers of diagonals
					spmv_dia_kernel_trans<value_type, index_type, BLOCK_SIZE, false> <<<grid, BLOCK_SIZE>>>
						(A.h(), A.w(),  A.num_dia(),  A.stride(),
						 A.get_offsets().ptr(), A.vec()->ptr(),
						 v.ptr(), dst.ptr());
				}
				cuvSafeCall(cudaThreadSynchronize());
			}

		template <bool transA, typename value_type, typename index_type>
			void spmv_dia_tex_device(const dev_dia_matrix<value_type,index_type>& A, 
					const dev_vector<value_type>& v, 
					dev_vector<value_type>& dst)
			{
				const unsigned int BLOCK_SIZE = 256;
				const dim3 grid = make_large_grid(A.h(),BLOCK_SIZE);

				cuvAssert(A.num_dia() < BLOCK_SIZE); // kernel doesn't handle larger numbers of diagonals

				bind_x(v.ptr());

				if(!transA){
					const unsigned int BLOCK_SIZE = 256;
					const dim3 grid = make_large_grid(A.h(),BLOCK_SIZE);
					cuvAssert(A.num_dia() < BLOCK_SIZE); // kernel doesn't handle larger numbers of diagonals
					spmv_dia_kernel2<value_type, index_type, BLOCK_SIZE, true> <<<grid, BLOCK_SIZE>>>
						(A.h(), A.w(),  A.num_dia(),  A.stride(),
						 A.get_offsets().ptr(), A.vec()->ptr(),
						 v.ptr(), dst.ptr());
				}else{
					const unsigned int BLOCK_SIZE = 256;
					const dim3 grid = make_large_grid(A.w(),BLOCK_SIZE);
					cuvAssert(A.num_dia() < BLOCK_SIZE); // kernel doesn't handle larger numbers of diagonals
					spmv_dia_kernel_trans<value_type, index_type, BLOCK_SIZE, true> <<<grid, BLOCK_SIZE>>>
						(A.h(), A.w(),  A.num_dia(),  A.stride(),
						 A.get_offsets().ptr(), A.vec()->ptr(),
						 v.ptr(), dst.ptr());
				}

				unbind_x(v.ptr());
			}
		template<class value_type, class index_type>
			void spmv(dev_vector<value_type,index_type>& dst, dev_dia_matrix<value_type,index_type>& A, dev_vector<value_type,index_type>& v, char transA, const float& factAv, const float& factC){
				spmv_dia_device(A,v,dst,transA,factAv,factC);
			}


		/****************************************************************
		 *  Host Code
		 ****************************************************************/
		template<class value_type, class index_type>
			void spmv(host_vector<value_type,index_type>& dst, host_dia_matrix<value_type,index_type>& A, host_vector<value_type,index_type>& v, char transA, const float& factAv, const float& factC){
				const host_vector<int>& offsets = A.get_offsets();
				const int num_diags             = A.num_dia();
				const int num_rows              = A.h();
				const int num_cols              = A.w();
				const int stride                = A.stride();
				cuvAssert(!A.transposed());
				if(transA == 't'){
					cuvAssert(num_rows == v.size());
					cuvAssert(num_cols == dst.size());
					for(index_type i = 0; i < num_diags; i++){
						const int k = offsets[i];  //diagonal offset

						const index_type i_start = std::max((int)0, k);
						const index_type j_start = std::max((int)0,-k);

						//number of elements to process
						const index_type N = std::min(num_rows - j_start, num_cols - i_start);

						const value_type * d_ = A.vec()->ptr() + i*stride + j_start;
						const value_type * x_ = v.ptr() + j_start;
						value_type * y_ = dst.ptr() + i_start;

						for(index_type n = 0; n < N; n++){
							y_[n] += d_[n] * x_[n];
						}
					}
				}else{
					cuvAssert(num_cols == v.size());
					cuvAssert(num_rows == dst.size());
					for(index_type i = 0; i < num_diags; i++){
						const int k = offsets[i];  //diagonal offset

						const index_type i_start = std::max((int)0,-k);
						const index_type j_start = std::max((int)0, k);

						//number of elements to process
						const index_type N = std::min(num_rows - i_start, num_cols - j_start);

						const value_type * d_ = A.vec()->ptr() + i*stride + i_start;
						const value_type * x_ = v.ptr() + j_start;
						value_type * y_ = dst.ptr() + i_start;

						for(index_type n = 0; n < N; n++){
							y_[n] += d_[n] * x_[n];
						}
					}
				}
			}
	}

	template<>
		void prod(host_dense_matrix<float,column_major>& dst,
				  host_dia_matrix<float>&                  A,
				  host_dense_matrix<float,column_major>&   B,
				  char transA,
				  char transB,
				  const float& factAB,
				  const float& factC){
			cuvAssert(transB == 'n');
			cuvAssert(dst.w() == B.w());
			for(int i=0;i<dst.w();i++){
				host_vector<float> dst_v(dst.h(), dst.vec().ptr()+i*dst.h(), true);
				host_vector<float> src_v(B.h(),   B.vec().ptr()+i*B.h(), true);
				spmv(dst_v,A,src_v,transA,factAB,factC);
			}
		}
	template<>
		void prod(dev_dense_matrix<float,column_major>& dst,
				  dev_dia_matrix<float>&                  A,
				  dev_dense_matrix<float,column_major>&   B,
				  char transA,
				  char transB,
				  const float& factAB,
				  const float& factC){
			cuvAssert(transB == 'n');
			cuvAssert(dst.w() == B.w());
			for(int i=0;i<dst.w();i++){
				dev_vector<float> dst_v(dst.h(), dst.vec().ptr()+i*dst.h(), true);
				dev_vector<float> src_v(B.h(),   B.vec().ptr()+i*B.h(), true);
				spmv(dst_v,A,src_v,transA,factAB,factC);
			}
		}
	template<class __matrix_type, class __vector_type>
		void spmv(__vector_type& dst, __matrix_type& A, __vector_type& v, char transA, const float& factAv, const float& factC){
			spmv_impl::spmv(dst,A,v,transA,factAv,factC);
		}
	template void spmv<host_dia_matrix<float>, host_vector<float> >(host_vector<float>&dst, host_dia_matrix<float>& A, host_vector<float>& v, char, const float&, const float&);
	template void spmv<dev_dia_matrix<float>, dev_vector<float> >(dev_vector<float>&dst, dev_dia_matrix<float>& A, dev_vector<float>& v, char, const float&, const float&);
}
