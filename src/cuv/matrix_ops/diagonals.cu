#include <memory>
#include <numeric>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/matrix_ops/diagonals.hpp>

namespace cuv{

	namespace avg_diagonals_impl{

		/***********************************************************
		 * With vector result
		 ***********************************************************/
		template<class T>
			void avg_diagonals(
					cuv::tensor<T,dev_memory_space>& dst,
					const cuv::dia_matrix<T,dev_memory_space>& dia
					){
				for( int i=0;i<dia.num_dia();i++ ){
					std::auto_ptr<const tensor<T,dev_memory_space> > diagonal ( dia.get_dia( dia.get_offset( i ) ));
					dst[i]= mean( *(diagonal.get()) );
				}
			}
		template<class T>
			void avg_diagonals(
					cuv::tensor<T,host_memory_space>& dst,
					const cuv::dia_matrix<T,host_memory_space>& dia
					){
				cuvAssert( dia.row_fact( )==1 );
				cuvAssert( dia.num_dia( )==dst.size( ));
				typedef unsigned int index_type;
				typedef T value_type;
				unsigned int A_stride = dia.stride();
				unsigned int A_h      = dia.h();
				unsigned int A_w      = dia.w();
				const cuv::tensor<int,host_memory_space>& offsets = dia.get_offsets();
				for( unsigned int i=0; i<dia.num_dia(); i++ ){
					T sum=0;
					const int k = offsets[i];  //diagonal offset
					const index_type i_start = std::max((int)0,-k);
					const index_type j_start = std::max((int)0, k);

					//number of elements to process
					const index_type N = std::min(A_h - i_start, (A_w - j_start));
					const value_type * d = dia.vec().ptr() + i*A_stride + i_start;
					for( int j=0;j<N;j++)
						sum += *d++;
					dst[i]= sum/N;
				}
			}
	}

	template<class T, class M>
	void avg_diagonals( cuv::tensor<T,M>& dst, const cuv::dia_matrix<T,M>& dia ){
		avg_diagonals_impl::avg_diagonals( dst, dia );
	}

	template void
	avg_diagonals(cuv::tensor<float,host_memory_space>&, const cuv::dia_matrix<float,host_memory_space>&);
	template void
	avg_diagonals(cuv::tensor<float,dev_memory_space>&, const cuv::dia_matrix<float,dev_memory_space>&);

}
