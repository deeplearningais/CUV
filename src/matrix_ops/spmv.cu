#include <iostream>
#include <boost/any.hpp>
#include <host_dia_matrix.hpp>
#include <dev_dia_matrix.hpp>
#include "matrix_ops.hpp"
#include <texture.h>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

using namespace std;

// stuff from NVIDIA SDK
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define small_grid_thread_id(void) ((__umul24(blockDim.x, blockIdx.x) + threadIdx.x))
#define large_grid_thread_id(void) ((__umul24(blockDim.x,blockIdx.x + __umul24(blockIdx.y,gridDim.x)) + threadIdx.x))
#define large_grid_thread_num(void) ((__umul24(blockDim.x,gridDim.x + __umul24(blockDim.y,gridDim.y))))

#define MAX_NUM_IMGS_AT_ONCE 6
/*#define MAX_NUM_IMGS_AT_ONCE 1*/


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
		template <typename value_type, typename index_type, unsigned int BLOCK_SIZE, bool UseCache, int NUM_IMG, bool wantFactAv, bool wantFactC>
			__global__ void
			spmm_dia_kernel_trans_register(const index_type A_h, 
					const index_type A_w, 
					const index_type A_nd,
					const index_type A_stride,
					const int        * A_diaoff,
					const value_type * A_data,
					const value_type * v, 
					value_type       * dst,
					const value_type factAv,
					const value_type factC)
			{
				__shared__ int        offsets[BLOCK_SIZE];
				value_type            sums[NUM_IMG];

				const index_type thread_id = large_grid_thread_id();
				const index_type grid_size = large_grid_thread_num();

				// load diagonal offsets into shared memory
				if(threadIdx.x < A_nd)
					offsets[threadIdx.x] = A_diaoff[threadIdx.x];
				__syncthreads();

				for(index_type col = thread_id; col < A_w; col += grid_size)
				{
					for(unsigned int i=0;i<NUM_IMG;i++)
						sums[i] = (value_type)0 ;
					index_type offset = 0;
					for(index_type n = 0; n < A_nd; n++, offset+=A_stride)
					{
						const int row = col - offsets[n];
						if(row >= 0 && row < A_h)
						{
							const value_type A_ij = A_data[       offset + row];
							for(unsigned int i=0;i<NUM_IMG;i++)
								sums[i] += A_ij * fetch_x<UseCache>(row,v + i * A_h);
						}
					}
					__syncthreads();
					for(unsigned int i=0;i<NUM_IMG;i++){
						dst[col + i*A_w] = (wantFactC  ? factC * dst[col + i * A_w] : 0.f) 
							+              (wantFactAv ? factAv                     : 1.f) * sums[i];
					}
				}
			}
		template <typename value_type, typename index_type, unsigned int BLOCK_SIZE, bool UseCache, int NUM_IMG, bool wantFactAv, bool wantFactC>
			__global__ void
			spmm_dia_kernel_trans_shared(const index_type A_h, 
					const index_type A_w, 
					const index_type A_nd,
					const index_type A_stride,
					const int        * A_diaoff,
					const value_type * A_data,
					const value_type * v, 
					value_type       * dst,
					const value_type factAv,
					const value_type factC)
			{
				__shared__ int        offsets[BLOCK_SIZE];
				__shared__ value_type    sums[BLOCK_SIZE * NUM_IMG];

				const index_type thread_id = large_grid_thread_id();
				const index_type grid_size = large_grid_thread_num();

				// load diagonal offsets into shared memory
				if(threadIdx.x < A_nd)
					offsets[threadIdx.x] = A_diaoff[threadIdx.x];
				__syncthreads();

				for(index_type col = thread_id; col < A_w; col += grid_size)
				{
					for(unsigned int i=0;i<NUM_IMG;i++)
						sums[BLOCK_SIZE*i + threadIdx.x] = (value_type)0 ;
					index_type offset = 0;
					for(index_type n = 0; n < A_nd; n++, offset+=A_stride)
					{
						const int row = col - offsets[n];
						if(row >= 0 && row < A_h)
						{
							const value_type A_ij = A_data[       offset + row];
							for(unsigned int i=0;i<NUM_IMG;i++)
								sums[BLOCK_SIZE*i + threadIdx.x] += A_ij * fetch_x<UseCache>(row,v + i * A_h);
						}
					}
					for(unsigned int i=0;i<NUM_IMG;i++){
						dst[col + i*A_w] = (wantFactC  ? factC * dst[col + i * A_w] : 0.f) 
							+              (wantFactAv ? factAv                     : 1.f) * sums[BLOCK_SIZE*i + threadIdx.x];
					}
				}
			}
		template <typename value_type, typename index_type, unsigned int BLOCK_SIZE, bool UseCache, unsigned int NUM_IMG, bool wantFactAv, bool wantFactC>
			__global__ void
			spmm_dia_kernel_register(
					const index_type A_h, 
					const index_type A_w, 
					const index_type A_nd,
					const index_type A_stride,
					const int        * A_diaoff,
					const value_type * A_data,
					const value_type * v, 
					value_type       * dst,
					const value_type factAv,
					const value_type factC)
			{
				__shared__ int        offsets[BLOCK_SIZE];
				value_type            sums[NUM_IMG];

				const index_type thread_id = large_grid_thread_id();
				const index_type grid_size = large_grid_thread_num();

				// load diagonal offsets into shared memory
				if(threadIdx.x < A_nd)
					offsets[threadIdx.x] = A_diaoff[threadIdx.x];

				for(index_type row = thread_id; row < A_h; row += grid_size)
				{
					// initialize shared memory
					for(unsigned int i=0;i<NUM_IMG;i++)
						sums[i] = (value_type) 0 ;
					__syncthreads();
					index_type offset = row;
					for(index_type n = 0; n < A_nd; n++, offset+=A_stride)
					{
						const int col = row + offsets[n];
						if(col >= 0 && col < A_w)
						{
							const value_type A_ij = A_data[       offset];
							for(unsigned int i=0;i<NUM_IMG;i++)
								sums[i] += A_ij * fetch_x<UseCache>(col,v + i * A_w);
						}
					}
					__syncthreads();
					for(unsigned int i=0;i<NUM_IMG;i++){
						dst[row + i*A_h] = (wantFactC  ? factC * dst[row + i * A_h] : 0.f) 
							+              (wantFactAv ? factAv                     : 1.f) * sums[i];
					}
				}
			}
		template <typename value_type, typename index_type, unsigned int BLOCK_SIZE, bool UseCache, unsigned int NUM_IMG, bool wantFactAv, bool wantFactC>
			__global__ void
			spmm_dia_kernel_shared(
					const index_type A_h, 
					const index_type A_w, 
					const index_type A_nd,
					const index_type A_stride,
					const int        * A_diaoff,
					const value_type * A_data,
					const value_type * v, 
					value_type       * dst,
					const value_type factAv,
					const value_type factC)
			{
				__shared__ int        offsets[BLOCK_SIZE];
				__shared__ value_type    sums[BLOCK_SIZE * NUM_IMG];

				const index_type thread_id = large_grid_thread_id();
				const index_type grid_size = large_grid_thread_num();

				// load diagonal offsets into shared memory
				if(threadIdx.x < A_nd)
					offsets[threadIdx.x] = A_diaoff[threadIdx.x];
				__syncthreads();

				for(index_type row = thread_id; row < A_h; row += grid_size)
				{
					// initialize shared memory
					for(unsigned int i=0;i<NUM_IMG;i++)
						sums[BLOCK_SIZE*i + threadIdx.x] = (value_type) 0 ;
					index_type offset = row;
					for(index_type n = 0; n < A_nd; n++, offset+=A_stride)
					{
						const int col = row + offsets[n];
						if(col >= 0 && col < A_w)
						{
							const value_type A_ij = A_data[       offset];
							for(unsigned int i=0;i<NUM_IMG;i++)
								sums[BLOCK_SIZE*i + threadIdx.x] += A_ij * fetch_x<UseCache>(col,v + i * A_w);
						}
					}
					for(unsigned int i=0;i<NUM_IMG;i++){
						dst[row + i*A_h] = (wantFactC  ? factC * dst[row + i * A_h] : 0.f) 
							+              (wantFactAv ? factAv                     : 1.f) * sums[BLOCK_SIZE*i + threadIdx.x];
					}
				}
			}
		/*
		 * General declaration of spmm_dia
		 */
		template <typename value_type, typename index_type, unsigned int BLOCK_SIZE, unsigned int NUM_IMG, bool transA>
			struct spmm_dia{
				static void apply(
						const dev_dia_matrix<value_type,index_type>& A,
						const dev_vector<value_type,index_type>& v,
						dev_vector<value_type,index_type>& dst,
						const float& factAv,
						const float& factC);
			};

		// spmm_dia implementation for non-transposed A
		template <typename value_type, typename index_type, unsigned int BLOCK_SIZE, unsigned int NUM_IMG>
			struct spmm_dia<value_type, index_type, BLOCK_SIZE, NUM_IMG, false>{
				static void apply(
						const dev_dia_matrix<value_type,index_type>& A,
						const dev_vector<value_type,index_type>& v,
						dev_vector<value_type,index_type>& dst,
						const float& factAv,
						const float& factC){
					const dim3 grid = make_large_grid(A.h(),BLOCK_SIZE);
					cuvAssert(A.num_dia() <= BLOCK_SIZE); // kernel doesn't handle larger numbers of diagonals
					static const bool useCache = false;
					if(useCache)
						bind_x(v.ptr(), v.size());
					if(0);
					else if(factAv==1.f && factC == 0.f){
						if(NUM_IMG==1)
							spmm_dia_kernel_register<value_type, index_type, BLOCK_SIZE,useCache,NUM_IMG,false,false> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr(), factAv,factC);
						else
							spmm_dia_kernel_shared<value_type, index_type, BLOCK_SIZE,useCache,NUM_IMG,false,false> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr(), factAv,factC);
					}else if(factAv==1.f && factC != 0.f){
						if(NUM_IMG==1)
							spmm_dia_kernel_register<value_type, index_type, BLOCK_SIZE,useCache,NUM_IMG,false,true> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr(), factAv,factC);
						else
							spmm_dia_kernel_shared<value_type, index_type, BLOCK_SIZE,useCache,NUM_IMG,false,true> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr(), factAv,factC);
					}else if(factAv!=1.f && factC == 0.f){
						if(NUM_IMG==1)
							spmm_dia_kernel_register<value_type, index_type, BLOCK_SIZE,useCache,NUM_IMG,true,false> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr(), factAv,factC);
						else
							spmm_dia_kernel_shared<value_type, index_type, BLOCK_SIZE,useCache,NUM_IMG,true,false> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr(), factAv,factC);
					}else{
						if(NUM_IMG==1)
							spmm_dia_kernel_register<value_type, index_type, BLOCK_SIZE,useCache,NUM_IMG,true,true> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr(), factAv,factC);
						else
							spmm_dia_kernel_shared<value_type, index_type, BLOCK_SIZE,useCache,NUM_IMG,true,true> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr(), factAv,factC);
					}
					if(useCache)
						unbind_x(v.ptr());
				}
			};
		// spmm_dia implementation for transposed A
		template <typename value_type, typename index_type, unsigned int BLOCK_SIZE, unsigned int NUM_IMG>
			struct spmm_dia<value_type, index_type, BLOCK_SIZE, NUM_IMG, true>{
				static void apply(
						const dev_dia_matrix<value_type,index_type>& A,
						const dev_vector<value_type,index_type>& v,
						dev_vector<value_type,index_type>& dst,
						const float& factAv,
						const float& factC){
					const dim3 grid = make_large_grid(A.w(),BLOCK_SIZE);
					cuvAssert(A.num_dia() <= BLOCK_SIZE); // kernel doesn't handle larger numbers of diagonals
					if(0);
					else if(factAv==1.f && factC == 0.f){
						if(NUM_IMG==1)
							spmm_dia_kernel_trans_register<value_type, index_type, BLOCK_SIZE, false,NUM_IMG,false,false> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr(), factAv,factC);
						else
							spmm_dia_kernel_trans_shared<value_type, index_type, BLOCK_SIZE, false,NUM_IMG,false,false> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr(), factAv,factC);
					}else if(factAv==1.f && factC != 0.f){
						if(NUM_IMG==1)
							spmm_dia_kernel_trans_register<value_type, index_type, BLOCK_SIZE, false,NUM_IMG,false,true> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr(), factAv,factC);
						else
							spmm_dia_kernel_trans_shared<value_type, index_type, BLOCK_SIZE, false,NUM_IMG,false,true> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr(), factAv,factC);
					}else if(factAv!=1.f && factC == 0.f){
						if(NUM_IMG==1)
							spmm_dia_kernel_trans_register<value_type, index_type, BLOCK_SIZE, false,NUM_IMG,true,false> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr(), factAv,factC);
						else
							spmm_dia_kernel_trans_shared<value_type, index_type, BLOCK_SIZE, false,NUM_IMG,true,false> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr(), factAv,factC);
					}else{
						if(NUM_IMG==1)
							spmm_dia_kernel_trans_register<value_type, index_type, BLOCK_SIZE, false,NUM_IMG,true,true> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr(), factAv,factC);
						else
							spmm_dia_kernel_trans_shared<value_type, index_type, BLOCK_SIZE, false,NUM_IMG,true,true> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr(), factAv,factC);
					}
				}
			};

#define SPMM_CASE(z,numimg,trans) case (numimg+1): \
		spmm_dia<value_type,index_type,((numimg>4)?128:128),(numimg+1),trans>::apply(A,v,dst,factAv,factC);break;

		template <typename value_type, typename index_type>
			void spmv_dia_device(const dev_dia_matrix<value_type,index_type>& A, 
					const dev_vector<value_type>& v, 
					dev_vector<value_type>& dst, 
					char transA,
					const value_type& factAv,
					const value_type& factC)
			{
				if(transA != 't'){
					switch( v.size() / A.w() ){
						BOOST_PP_REPEAT(MAX_NUM_IMGS_AT_ONCE,SPMM_CASE,false); // image-num 1..32; not transposed
						default: 
						cout << "Not implemented: SPMM w/ num_img="<<v.size()/A.w()<<endl;
						cuvAssert(false);
					}
				}else{
					switch( v.size() / A.h() ){
						BOOST_PP_REPEAT(MAX_NUM_IMGS_AT_ONCE,SPMM_CASE,true);  // image-num 1..32;     transposed
						default: 
						cout << "Not implemented: SPMM w/ num_img="<<v.size()/A.h()<<endl;
						cuvAssert(false);
					}
				}
				cuvSafeCall(cudaThreadSynchronize());
			}

		/*template <bool transA, typename value_type, typename index_type>*/
		/*    void spmv_dia_tex_device(const dev_dia_matrix<value_type,index_type>& A, */
		/*            const dev_vector<value_type>& v, */
		/*            dev_vector<value_type>& dst)*/
		/*    {*/
		/*        const unsigned int BLOCK_SIZE = 256;*/
		/*        const dim3 grid = make_large_grid(A.h(),BLOCK_SIZE);*/

		/*        cuvAssert(A.num_dia() < BLOCK_SIZE); // kernel doesn't handle larger numbers of diagonals*/

		/*        bind_x(v.ptr());*/

		/*        if(!transA){*/
		/*            const unsigned int BLOCK_SIZE = 256;*/
		/*            const dim3 grid = make_large_grid(A.h(),BLOCK_SIZE);*/
		/*            cuvAssert(A.num_dia() < BLOCK_SIZE); // kernel doesn't handle larger numbers of diagonals*/
		/*            spmv_dia_kernel<value_type, index_type, BLOCK_SIZE, true> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr());*/
		/*        }else{*/
		/*            const unsigned int BLOCK_SIZE = 256;*/
		/*            const dim3 grid = make_large_grid(A.w(),BLOCK_SIZE);*/
		/*            cuvAssert(A.num_dia() < BLOCK_SIZE); // kernel doesn't handle larger numbers of diagonals*/
		/*            spmv_dia_kernel_trans<value_type, index_type, BLOCK_SIZE, true> <<<grid, BLOCK_SIZE>>> (A.h(), A.w(),  A.num_dia(),  A.stride(), A.get_offsets().ptr(), A.vec()->ptr(), v.ptr(), dst.ptr());*/
		/*        }*/

		/*        unbind_x(v.ptr());*/
		/*    }*/
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
				const int A_h                   = A.h();
				const int A_w                   = A.w();
				const int A_stride              = A.stride();
				cuvAssert(!A.transposed());
				index_type max_dst = ((transA=='t') ? A_w : A_h);
				if(factC==0.f)
					for(int i=0;i<max_dst;i++) dst[i] = 0;
				else
					for(int i=0;i<max_dst;i++) dst[i] = dst[i] * factC;
				if(transA == 't'){
					cuvAssert(A_h == v.size());
					cuvAssert(A_w == dst.size());
					for(index_type i = 0; i < num_diags; i++){
						const int k = offsets[i];  //diagonal offset

						const index_type i_start = std::max((int)0, k);
						const index_type j_start = std::max((int)0,-k);

						//number of elements to process
						const index_type N = std::min(A_h - j_start, A_w - i_start);

						const value_type * d_ = A.vec()->ptr() + i*A_stride + j_start;
						const value_type * x_ = v.ptr() + j_start;
						value_type * y_ = dst.ptr() + i_start;

						for(index_type n = 0; n < N; n++){
							y_[n] += factAv * d_[n] * x_[n];
						}
					}
				}else{
					cuvAssert(A_w == v.size());
					cuvAssert(A_h == dst.size());
					for(index_type i = 0; i < num_diags; i++){
						const int k = offsets[i];  //diagonal offset

						const index_type i_start = std::max((int)0,-k);
						const index_type j_start = std::max((int)0, k);

						//number of elements to process
						const index_type N = std::min(A_h - i_start, A_w - j_start);

						const value_type * d_ = A.vec()->ptr() + i*A_stride + i_start;
						const value_type * x_ = v.ptr() + j_start;
						value_type * y_ = dst.ptr() + i_start;

						for(index_type n = 0; n < N; n++){
							y_[n] += factAv * d_[n] * x_[n];
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
			const int num_at_same_time = min(MAX_NUM_IMGS_AT_ONCE, B.w());
			for(int i=0; i<dst.w(); i += num_at_same_time){
				dev_vector<float> dst_v(dst.h() * min(dst.w()-i,num_at_same_time), dst.vec().ptr()+i*dst.h(), true);
				dev_vector<float> src_v(B.h()   * min(B.w()-i,  num_at_same_time), B.vec().ptr()+i*B.h(), true);
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
