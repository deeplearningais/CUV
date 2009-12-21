#include <cublas.h>
#include <cblas.h>

#include <cuv_general.hpp>
#include "matrix_ops.hpp"

#define CVT_TRANSPOSE(c) \
	(CBLAS_TRANSPOSE)(((c) == 'N' || (c) == 'n') ? CblasNoTrans : \
	 ((c) == 'T' || (c) == 't') ? CblasTrans : \
	 ((c) == 'C' || (c) == 'c') ? CblasConjTrans : \
	 -1)



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

			bool res_is_vec = (dst.w() == 1 || dst.h() == 1);
			if(!res_is_vec){
				cuvAssert(dst.h() == m);
				cuvAssert(dst.w() == n);
			} 
			cuvAssert(k1 == k2);
			cuvAssert(A.ptr());
			cuvAssert(B.ptr());
			cuvAssert(dst.ptr());

			cublasSgemm(transA, transB, m, n, k1, factAB, A.ptr(), A.h(),B.ptr(), B.h(), factC, dst.ptr(), res_is_vec ? dst.n() : dst.h());
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

			bool res_is_vec = (dst.w() == 1 || dst.h() == 1);
			if(!res_is_vec){
				cuvAssert(dst.h() == m);
				cuvAssert(dst.w() == n);
			} 
			cuvAssert(k1 == k2);
			cuvAssert(A.ptr() != NULL);
			cuvAssert(B.ptr() != NULL);
			cuvAssert(dst.ptr());

#if 1 /* CBLAS */
			cblas_sgemm(
				   CblasColMajor,
				   CVT_TRANSPOSE(transA),
				   CVT_TRANSPOSE(transB), m, n, k1,
				   factAB, A.ptr(), A.h(),B.ptr(), B.h(), factC, dst.ptr(), res_is_vec ? dst.n() : dst.h());
#else /* naive */
			for(int i=0; i<A.h();i++)
				for(int j=0; j<B.w(); j++){
					float f=0;
					for(int k=0;k<A.w();k++){
						f += A(i,k)*B(k,j);
					}
					dst(i,j) = f;
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

			bool res_is_vec = (dst.w() == 1 || dst.h() == 1);
			if(!res_is_vec){
				cuvAssert(dst.h() == m);
				cuvAssert(dst.w() == n);
			} 
			cuvAssert(k1 == k2);
			cuvAssert(A.ptr() != NULL);
			cuvAssert(B.ptr() != NULL);
			cuvAssert(dst.ptr());

			cblas_sgemm(
					CblasRowMajor,
					CVT_TRANSPOSE(transA),
					CVT_TRANSPOSE(transB), m, n, k1, 
					factAB, A.ptr(), A.h(),B.ptr(), B.h(), factC, dst.ptr(), res_is_vec ? dst.n() : dst.h());
		}


}; // cuv
