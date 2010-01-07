#include <host_dia_matrix.hpp>
#include <dev_dia_matrix.hpp>
#include "matrix_ops.hpp"

namespace cuv{
	namespace spmv_impl{
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
	template<class __matrix_type, class __vector_type>
		void spmv(__vector_type& dst, __matrix_type& A, __vector_type& v, char transA, const float& factAv, const float& factC){
			spmv_impl::spmv(dst,A,v,transA,factAv,factC);
		}
	template void spmv<host_dia_matrix<float>, host_vector<float> >(host_vector<float>&dst, host_dia_matrix<float>& A, host_vector<float>& v, char, const float&, const float&);
}
