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
void reduce_to_col_kernel(const T* matrix, T* vector, int nCols, int nRows, T param, T factNew, T factOld) {
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
		vector[bx * blockDim.x + tx] = vector[bx*blockDim.x+tx]*factOld + shared[0][tx] * factNew;
	__syncthreads();
}

template<int BLOCK_SIZE, class T, int OP>
__global__
void reduce_to_row_kernel(const T* matrix, T* vector, int nCols, int nRows, T param, T factNew, T factOld) {
    __shared__ T shared[BLOCK_SIZE*BLOCK_SIZE];
	int tx = threadIdx.x, bx=blockIdx.x;
	int ty = threadIdx.y, by=blockIdx.y;
	if(by*blockDim.y + ty>nCols) return;
	int off = blockDim.x;

	shared[tx] = 0.f;
	for(int my=tx; my<nRows; my += off){
		T f = matrix[by * nRows + bx*blockDim.x + my];

		if(OP==1)  f*= f;
		if(OP==2)  f*= param;
		if(OP==3)  f= -log(f);
		shared[tx] +=f;
	}
	__syncthreads();

	int offset = blockDim.x / 2;
	while(offset > 0) {
		if( tx < offset)
			shared[tx] += shared[tx+offset];
		offset >>= 1;
		__syncthreads();
	}

	if (tx == 0)
		vector[by * blockDim.y + ty] = vector[by * blockDim.y + ty] * factOld + shared[0] * factNew;
	__syncthreads();
}


template<int BLOCK_SIZE, class T, class I>
__global__
void argmax_row_kernel(const T* matrix, I* vector, int nCols, int nRows) {
	__shared__ I shIdx[BLOCK_SIZE*BLOCK_SIZE]; // index of the maximum
	__shared__ T shVal[BLOCK_SIZE*BLOCK_SIZE]; // value

	int tx = threadIdx.x, bx=blockIdx.x;
	int ty = threadIdx.y, by=blockIdx.y;
	if(by*blockDim.y + ty>nCols) return;
	int off = blockDim.x;

	int idx = by * nRows + bx*blockDim.x + tx;
	shVal[tx] = matrix[idx];
	shIdx[tx] = tx;

	for(int my=tx+off; my<nRows; my += off){
		idx += off;
		T f = matrix[idx];

		if(f > shVal[tx]) {
			shVal[tx] = f;
			shIdx[tx] = my;
		}
	}
	__syncthreads();

	int offset = blockDim.x / 2;
	while(offset > 0) {
		if( tx < offset) {
			if(shVal[tx] < shVal[tx+offset]) {
				shVal[tx] = shVal[tx+offset];
				shIdx[tx] = shIdx[tx+offset];
			}
		}
		offset >>= 1;
		__syncthreads();
	}

	if (tx == 0)
		vector[by * blockDim.y + ty] = shIdx[0];
	__syncthreads();
}

// "coalesced transpose" with no bank conflicts, example from SDK
// potential speedup by 5 possible for "fine-grained transpose"
template<int BLOCK_SIZE, class T>
__global__
void transpose_kernel(T* dst, T* src, int width, int height) {
    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;
    const int tx = bx + threadIdx.x;
    const int ty = by + threadIdx.y;

    __shared__ T shared[BLOCK_SIZE][BLOCK_SIZE + 1];

    if (tx < width && ty < height) {
        shared[threadIdx.y][threadIdx.x] = src[ty * width + tx];
    }
    __syncthreads();

    if (by + threadIdx.x < height && threadIdx.y + bx < width) {
        dst[(bx + threadIdx.y) * height + by + threadIdx.x] = shared[threadIdx.x][threadIdx.y];
    }
}


namespace cuv{
	template<>
		host_dense_matrix<float,column_major>* blockview(
				host_dense_matrix<float,column_major>& matrix,
				unsigned int start_rows,
				unsigned int num_rows,
				unsigned int start_cols,
				unsigned int num_cols) {
			cuvAssert(start_rows==0);
			cuvAssert(num_rows==matrix.h())
			return new host_dense_matrix<float,column_major>(num_rows,num_cols, matrix.ptr()+matrix.h()*start_cols,true);
		}
	template<>
		dev_dense_matrix<float,column_major>* blockview(
				dev_dense_matrix<float,column_major>& matrix,
				unsigned int start_rows,
				unsigned int num_rows,
				unsigned int start_cols,
				unsigned int num_cols) {
			cuvAssert(start_rows==0);
			cuvAssert(num_rows==matrix.h())
			return new dev_dense_matrix<float,column_major>(num_rows,num_cols, matrix.ptr()+matrix.h()*start_cols,true);
		}

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
	  void matrix_divide_col(__matrix_type& A, const __vector_type& v){
		  matrix_plus_vector_impl::matrix_plus_col(A,v, thrust::divides<typename __matrix_type::value_type>());
	  }

  namespace reduce_to_col_impl{
	  template<class V,class I, class V2>
	  void reduce_to_col(host_vector<V2,I>&v, const host_dense_matrix<V,column_major,I>& m, const V& factNew, const V& factOld){
		  cuvAssert(m.ptr() != NULL);
		  cuvAssert(m.h()   == v.size());
		  const  V* A_ptr = m.ptr();
		  
		  V2* v_ptr = v.ptr();
		  for(int j=0; j<m.h(); j++){
				  *v_ptr++  *= factOld;
		  }
		  for(int i=0;i<m.w();i++){
			  v_ptr = v.ptr();
			  for(int j=0; j<m.h(); j++){
				  *v_ptr++ += factNew * *A_ptr++;
			  }
		  }
	  }
	  template<class V,class I, class V2>
	  void reduce_to_col(host_vector<V2,I>&v, const host_dense_matrix<V,row_major,I>& m, const V& factNew, const V& factOld){
		  cuvAssert(m.ptr() != NULL);
		  cuvAssert(m.h()   == v.size());
		  const  V* A_ptr = m.ptr();

		  V2* v_ptr = v.ptr();
		  for(int j=0; j<m.h(); j++){
				  *v_ptr++  *= factOld;
		  }
		  v_ptr = v.ptr();
		  for(int i=0;i<m.h(); i++) {
			 for(int j=0; j<m.w(); j++)
				 *v_ptr += factNew * *A_ptr++;
			  v_ptr++;
		  }
	  }
	  template<class V,class I, class V2>
	  void reduce_to_col(dev_vector<V2,I>&v, const dev_dense_matrix<V,column_major,I>& m, const V& factNew, const V& factOld){
		  cuvAssert(m.ptr() != NULL);
		  cuvAssert(m.h()   == v.size());
		  static const int BLOCK_SIZE = 16;
		  dim3 grid(ceil((float)m.h()/(BLOCK_SIZE/2)), 1);
		  dim3 threads(BLOCK_SIZE/2,BLOCK_SIZE*2);
		  reduce_to_col_kernel<BLOCK_SIZE,V,0><<<grid,threads>>>(m.ptr(),v.ptr(),m.w(),m.h(),0,factNew,factOld);
		  cuvSafeCall(cudaThreadSynchronize());
	  }
	template<class V,class I, class V2>
	void reduce_to_col(dev_vector<V2,I>&v, const dev_dense_matrix<V,row_major,I>& m, const V& factNew, const V& factOld){
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.h() == v.size());
		static const int BLOCK_SIZE = 16;
		dim3 grid(1, m.h());
		dim3 threads(BLOCK_SIZE*BLOCK_SIZE,1);
		// yes, we abuse the reduce_to_row kernel here :)
		reduce_to_row_kernel<BLOCK_SIZE,V,0><<<grid,threads>>>(m.ptr(),v.ptr(),m.h(),m.w(),0,factNew,factOld);
		cuvSafeCall(cudaThreadSynchronize());
	}

  }
  template<class __matrix_type, class __vector_type>
	  void reduce_to_col(__vector_type&v, const __matrix_type& m, const typename __matrix_type::value_type& factNew, const typename __matrix_type::value_type& factOld){
		  reduce_to_col_impl::reduce_to_col(v,m,factNew,factOld);
	  }

  namespace reduce_to_row_impl{
	  template<class V,class I, class V2>
	  void reduce_to_row(host_vector<V2,I>&v, const host_dense_matrix<V,row_major,I>& m, const V& factNew, const V& factOld){
		  cuvAssert(m.ptr() != NULL);
		  cuvAssert(m.w()   == v.size());
		  const  V* A_ptr = m.ptr();

		  V2* v_ptr = v.ptr();
		  for(int j=0; j<v.size(); j++){
				  *v_ptr++  *= factOld;
		  }
		  for(int i=0;i<m.h();i++){
			  v_ptr = v.ptr();
			  for(int j=0; j<m.w(); j++){
				  *v_ptr++ += factNew * *A_ptr++;
			  }
		  }
	  }
	  template<class V,class I, class V2>
	  void reduce_to_row(host_vector<V2,I>&v, const host_dense_matrix<V,column_major,I>& m, const V& factNew, const V& factOld) {
		  cuvAssert(m.ptr() != NULL);
		  cuvAssert(m.w()   == v.size());
		  const  V* A_ptr = m.ptr();

		  V2* v_ptr = v.ptr();
		  for(int j=0; j<v.size(); j++){
				  *v_ptr++  *= factOld;
		  }
		  v_ptr = v.ptr();
		  for(int i=0;i<m.w();i++){
			  for(int j=0; j<m.h(); j++)
				  *v_ptr += factNew * *A_ptr++;
			  v_ptr++;
		  }
	  }

	template<class V,class I, class V2>
	void reduce_to_row(dev_vector<V2,I>&v, const dev_dense_matrix<V,column_major,I>& m, const V& factNew, const V& factOld){
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.w() == v.size());
		static const int BLOCK_SIZE = 16;
		dim3 grid(1, m.w());
		dim3 threads(BLOCK_SIZE*BLOCK_SIZE,1);
		reduce_to_row_kernel<BLOCK_SIZE,V,0><<<grid,threads>>>(m.ptr(),v.ptr(),m.w(),m.h(),0,factNew,factOld);
		cuvSafeCall(cudaThreadSynchronize());
	}
	template<class V,class I, class V2>
		void reduce_to_row(dev_vector<V2,I>&v, const dev_dense_matrix<V,row_major,I>& m, const V& factNew, const V& factOld){
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.w()   == v.size());
		static const int BLOCK_SIZE = 16;
		dim3 grid(ceil((float)m.w()/(BLOCK_SIZE/2)), 1);
		dim3 threads(BLOCK_SIZE/2,BLOCK_SIZE*2);
		// yes, we abuse the reduce_to_col kernel here :)
		reduce_to_col_kernel<BLOCK_SIZE,V,0><<<grid,threads>>>(m.ptr(),v.ptr(),m.h(),m.w(),0,factNew,factOld);
		cuvSafeCall(cudaThreadSynchronize());
	}

  }
  template<class __matrix_type, class __vector_type>
	  void reduce_to_row(__vector_type&v, const __matrix_type& m, const typename __matrix_type::value_type& factNew, const typename __matrix_type::value_type& factOld){
		  reduce_to_row_impl::reduce_to_row(v,m,factNew,factOld);
	  }

	template<>
	void argmax_to_row(dev_vector<int>&v, const dev_dense_matrix<float,column_major>& m){
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.w() == v.size());
		static const int BLOCK_SIZE = 16;
		dim3 grid(1, m.w());
		dim3 threads(BLOCK_SIZE*BLOCK_SIZE,1);
		argmax_row_kernel<BLOCK_SIZE,float,int><<<grid,threads>>>(m.ptr(),v.ptr(),m.w(),m.h());
		cuvSafeCall(cudaThreadSynchronize());
	}

	template<>
	void argmax_to_column(dev_vector<int>&v, const dev_dense_matrix<float,row_major>& m){
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.h() == v.size());
		static const int BLOCK_SIZE = 16;
		dim3 grid(1, m.h());
		dim3 threads(BLOCK_SIZE*BLOCK_SIZE,1);
		argmax_row_kernel<BLOCK_SIZE,float,int><<<grid,threads>>>(m.ptr(),v.ptr(),m.h(),m.w());
		cuvSafeCall(cudaThreadSynchronize());
	}

	template<>
	void argmax_to_row(host_vector<int>&v, const host_dense_matrix<float,column_major>& m){
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.w() == v.size());
		const float* ptr = m.ptr();
		int* res = v.ptr();
		for(int i=0; i<m.w();i++) {
			int idx = 0;
			float val = *ptr;
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

	template<>
	void argmax_to_column(host_vector<int>&v, const host_dense_matrix<float,row_major>& m){
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.h() == v.size());
		const float* ptr = m.ptr();
		int* res = v.ptr();
		for(int i=0; i<m.h();i++) {
			int idx = 0;
			float val = *ptr;
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

	template<>
	void transpose(dev_dense_matrix<float,column_major>& dst,
		dev_dense_matrix<float,column_major>& src) {
		cuvAssert(dst.w() == src.h());
		cuvAssert(dst.h() == src.w());
	    const int width = dst.w();
	    const int height = dst.h();
		static const int BLOCK_SIZE = 16;
	    const int numBlocksX = ceil((float)width / BLOCK_SIZE);
	    const int numBlocksY = ceil((float)height / BLOCK_SIZE);
	    dim3 gridSize(numBlocksX, numBlocksY, 1);
	    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
	    transpose_kernel<BLOCK_SIZE,float><<<gridSize, blockSize>>>(dst.ptr(), src.ptr(), width, height);
		cuvSafeCall(cudaThreadSynchronize());
	}

	template<>
	void transpose(dev_dense_matrix<float,row_major>& dst,
		dev_dense_matrix<float,row_major>& src) {
		cuvAssert(dst.w() == src.h());
		cuvAssert(dst.h() == src.w());
	    const int width = dst.h();
	    const int height = dst.w();
		static const int BLOCK_SIZE = 16;
	    const int numBlocksX = ceil((float)width / BLOCK_SIZE);
	    const int numBlocksY = ceil((float)height / BLOCK_SIZE);
	    dim3 gridSize(numBlocksX, numBlocksY, 1);
	    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
	    transpose_kernel<BLOCK_SIZE,float><<<gridSize, blockSize>>>(dst.ptr(), src.ptr(), width, height);
		cuvSafeCall(cudaThreadSynchronize());
	}

	template<>
	void transpose(host_dense_matrix<float,column_major>& dst,
		host_dense_matrix<float,column_major>& src) {
		cuvAssert(dst.w() == src.h());
		cuvAssert(dst.h() == src.w());
		float* dst_ptr = dst.ptr();
		float* src_ptr = src.ptr();
		for(int i=0; i<dst.w(); i++) {
			for(int j=0; j<dst.h(); j++) {
				*dst_ptr++ = src_ptr[j*src.h()];
			}
			src_ptr++;
		}
	}

	template<>
	void transpose(host_dense_matrix<float,row_major>& dst,
		host_dense_matrix<float,row_major>& src) {
		cuvAssert(dst.w() == src.h());
		cuvAssert(dst.h() == src.w());
		float* dst_ptr = dst.ptr();
		float* src_ptr = src.ptr();
		for(int i=0; i<dst.h(); i++) {
			for(int j=0; j<dst.w(); j++) {
				*dst_ptr++ = src_ptr[j*src.w()];
			}
			src_ptr++;
		}
	}


#define INSTANTIATE_MV(V,M) \
  template void matrix_plus_col(dev_dense_matrix<V,M>&, const dev_vector<V>&);   \
  template void matrix_plus_col(host_dense_matrix<V,M>&, const host_vector<V>&); \
  template void matrix_times_col(dev_dense_matrix<V,M>&, const dev_vector<V>&);  \
  template void matrix_times_col(host_dense_matrix<V,M>&, const host_vector<V>&); \
  template void matrix_divide_col(dev_dense_matrix<V,M>&, const dev_vector<V>&);  \
  template void matrix_divide_col(host_dense_matrix<V,M>&, const host_vector<V>&);

#define INSTANTIATE_REDCOL(V,M) \
  template void reduce_to_row(dev_vector<V>&, const dev_dense_matrix<V,M>&, const V&,const V&); \
  template void reduce_to_col(dev_vector<V>&, const dev_dense_matrix<V,M>&, const V&,const V&); \
  template void reduce_to_row(host_vector<V>&, const host_dense_matrix<V,M>&, const V&,const V&); \
  template void reduce_to_col(host_vector<V>&, const host_dense_matrix<V,M>&, const V&,const V&);

#define INSTANTIATE_REDROW(V,M) \
  template void reduce_to_col(dev_vector<V>&, const dev_dense_matrix<V,M>&, const V&,const V&); \
  template void reduce_to_row(dev_vector<V>&, const dev_dense_matrix<V,M>&, const V&,const V&); \
  template void reduce_to_col(host_vector<V>&, const host_dense_matrix<V,M>&, const V&,const V&); \
  template void reduce_to_row(host_vector<V>&, const host_dense_matrix<V,M>&, const V&,const V&);

  INSTANTIATE_MV(float, column_major);
  INSTANTIATE_MV(float, row_major);

  INSTANTIATE_REDCOL(float,column_major);
  INSTANTIATE_REDROW(float,row_major);
}; // cuv
