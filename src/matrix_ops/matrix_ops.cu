#include <cublas.h>
#include <cblas.h>

#include <cuv_general.hpp>
#include <thrust/functional.h>
#include "matrix_ops.hpp"

#define CVT_TRANSPOSE(c) \
	(CBLAS_TRANSPOSE)(((c) == 'N' || (c) == 'n') ? CblasNoTrans : \
	 ((c) == 'T' || (c) == 't') ? CblasTrans : \
	 ((c) == 'C' || (c) == 'c') ? CblasConjTrans : \
	 -1)

template<int BLOCK_SIZE, class T, int OP>
__global__ 
void reduce_to_col_kernel(T* matrix, T* vector, int nCols, int nRows, T param) {
	__shared__ T shared[BLOCK_SIZE*2][BLOCK_SIZE/2];
	int tx = threadIdx.x, bx=blockIdx.x;
	int ty = threadIdx.y;//, by=blockIdx.y;
	if(bx*blockDim.x+tx>nRows) return;
	int off = blockDim.y;
	
	shared[ty][tx] = 0.f;
	for(int my=ty; my<nCols; my += off){
		T f = matrix[my*nRows + bx*blockDim.x + tx];

		if(OP==1)  f*= f;
		if(OP==2)  f*= param;
		shared[ty][tx] +=f;
	}
	__syncthreads();

	int offset = blockDim.y / 2;
	while(offset > 0) {
		if( ty < offset)
			shared[ty][tx] += shared[ty+offset][tx];
		offset >>= 1;
		__syncthreads();
	}

	if (ty == 0)
		vector[bx * blockDim.x + tx] = shared[0][tx];
	__syncthreads();
}


namespace cuv{

	template<>
		void prod(dev_dense_matrix<float,column_major>& dst,
				  dev_dense_matrix<float,column_major>&   A,
				  dev_dense_matrix<float,column_major>&   B,
				  char transA,
				  char transB,
				  const float& factAB,
				  const float& factC){
			int m  = (transA=='t' ? A.w() : A.h());
			int k1 = (transA=='t' ? A.h() : A.w());
			int k2 = (transB=='t' ? B.w() : B.h());
			int n  = (transB=='t' ? B.h() : B.w());

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
		void prod(host_dense_matrix<float,column_major>& dst,
				  host_dense_matrix<float,column_major>&   A,
				  host_dense_matrix<float,column_major>&   B,
				  char transA,
				  char transB,
				  const float& factAB,
				  const float& factC){
			int m  = (transA=='t' ? A.w() : A.h());
			int k1 = (transA=='t' ? A.h() : A.w());
			int k2 = (transB=='t' ? B.w() : B.h());
			int n  = (transB=='t' ? B.h() : B.w());

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
				for(int j=0; j<B.w(); j++){
					float f=0;
					for(int k=0;k<A.w();k++){
						f += A(i,k)*B(k,j);
					}
					dst.set(i,j,f);
				}
#endif
		}

	template<>
		void prod(host_dense_matrix<float,row_major>& dst,
				  host_dense_matrix<float,row_major>&   A,
				  host_dense_matrix<float,row_major>&   B,
				  char transA,
				  char transB,
				  const float& factAB,
				  const float& factC){
			int m  = (transA=='t' ? A.w() : A.h());
			int k1 = (transA=='t' ? A.h() : A.w());
			int k2 = (transB=='t' ? B.w() : B.h());
			int n  = (transB=='t' ? B.h() : B.w());

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
					factAB, A.ptr(), A.h(),B.ptr(), B.h(), factC, dst.ptr(), dst.h());
		}

	template<class V, class I, class V2, class OP>
	__global__
	void matrix_plus_vector_kernel_column_major(V*A,V2* v,I w,I h, OP op){
		int tid = blockDim.x*blockIdx.x + threadIdx.x;
		if(tid>h) return;
		V2 tid_v = v[tid];
		for(int i=tid;i<w;i++)
			A[i] = op(A[i],tid_v);
	}
	template<class V, class I, class V2, class OP>
	__global__ 
	void matrix_plus_vector_kernel_column_major2 (V *A, const V2* v,  I h, I w, OP op) {
			const unsigned int idx        = __mul24(blockIdx.x , blockDim.x) + threadIdx.x;
			const unsigned int numThreads = __mul24(blockDim.x , gridDim.x);

			int stop = w*h;
			for (unsigned int i = idx; i < stop; i += numThreads)
				A[i] = op(A[i] , v[i % h]);
		}
	template<class V, class I, class V2, class OP>
	__global__ 
	void matrix_plus_vector_kernel_row_major (V *A, V2* v,  I h, I w, OP op) {
			__shared__ float scalar;
			if (threadIdx.x == 0) {
				scalar = v[blockIdx.x];
			}
			__syncthreads();
			int stop = h;
			for (unsigned int i = threadIdx.x; i < stop; i += blockDim.x) {
				A[blockIdx.x * h + i] = op(A[blockIdx.x * h + i] , scalar);
			}
		}

	namespace matrix_plus_vector_impl{
		template<class V, class I, class V2, class OP>
			void matrix_plus_col(dev_dense_matrix<V,row_major,I>& A, const dev_vector<V2,I>& v, const OP& op){
				cuvAssert(A.h() == v.size());
				int num_threads = min(512,A.h());
				int num_blocks  = A.w();
				matrix_plus_vector_kernel_row_major<<<num_blocks,num_threads>>>(A.ptr(), v.ptr(), A.h(), A.w(), op);
				cuvSafeCall(cudaThreadSynchronize());
			}
		template<class V, class I, class V2, class OP>
			void matrix_plus_col(dev_dense_matrix<V,column_major,I>& A, const dev_vector<V2,I>& v, const OP& op){
				cuvAssert(A.h() == v.size());
				int num_threads = 512;
				int num_blocks  = min(512,(int)ceil((float)A.n() / num_threads));
				matrix_plus_vector_kernel_column_major2<<<num_blocks,num_threads>>>(A.ptr(), v.ptr(), A.h(), A.w(), op);
				cuvSafeCall(cudaThreadSynchronize());
			}
		template<class V, class I, class V2, class OP>
			void matrix_plus_col(host_dense_matrix<V,column_major,I>& A, const host_vector<V2,I>& v, const OP& op){
				const V2* v_ptr = v.ptr();
				V *       A_ptr = A.ptr();
				for(int j=0;j<A.w();j++){
					v_ptr = v.ptr();
					for(int i=0;i<A.h();i++,A_ptr++,v_ptr++)
						*A_ptr = op(*A_ptr,*v_ptr);
				}
			}
		template<class V, class I, class V2, class OP>
			void matrix_plus_col(host_dense_matrix<V,row_major,I>& A, const host_vector<V2,I>& v, const OP& op){
				const V2* v_ptr = v.ptr();
				V *       A_ptr = A.ptr();
				for(int i=0;i<A.h();i++, v_ptr++){
					for(int j=0;j<A.w();j++)
						*A_ptr++ = op(*A_ptr,*v_ptr);
				}
			}
	}
  template<class __matrix_type, class __vector_type>
	  void matrix_plus_col(__matrix_type& A, const __vector_type& v){
		  matrix_plus_vector_impl::matrix_plus_col(A,v, thrust::plus<typename __matrix_type::value_type>());
	  }

  template<class __matrix_type, class __vector_type>
	  void matrix_times_col(__matrix_type& A, const __vector_type& v){
		  matrix_plus_vector_impl::matrix_plus_col(A,v, thrust::multiplies<typename __matrix_type::value_type>());
	  }

  template<class __matrix_type, class __vector_type>
	  void reduce_to_col(__vector_type&v, const __matrix_type& m){
		  assert(m.ptr() != NULL);
		  assert(m.h()   == v.size());
		  fill(v,0);
		  static const int BLOCK_SIZE = 16;
		  dim3 grid(ceil((float)m.h()/(BLOCK_SIZE/2)), 1);
		  dim3 threads(BLOCK_SIZE/2,BLOCK_SIZE*2);
		  reduce_to_col_kernel<BLOCK_SIZE,typename __matrix_type::value_type,0><<<grid,threads>>>(m.ptr(),v.ptr(),m.w(),m.h(),0);
		  cuvSafeCall(cudaThreadSynchronize());
	  }

#define INSTANTIATE_MV(V,M) \
  template void matrix_plus_col(dev_dense_matrix<V,M>&, const dev_vector<V>&);   \
  template void matrix_plus_col(host_dense_matrix<V,M>&, const host_vector<V>&); \
  template void matrix_times_col(dev_dense_matrix<V,M>&, const dev_vector<V>&);  \
  template void matrix_times_col(host_dense_matrix<V,M>&, const host_vector<V>&);

  INSTANTIATE_MV(float, column_major);
  INSTANTIATE_MV(float, row_major);

}; // cuv
