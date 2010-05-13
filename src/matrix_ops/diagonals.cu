#include <memory>
#include <numeric>
#include "vector_ops/vector_ops.hpp"
#include "diagonals.hpp"

namespace cuv{

	namespace avg_diagonals_impl{

		/***********************************************************
		 * With vector result
		 ***********************************************************/
		template<class T,class I>
			void avg_diagonals(
					cuv::vector<T,dev_memory_space,I>& dst,
					const cuv::dia_matrix<T,dev_memory_space,I>& dia
					){
				for( int i=0;i<dia.num_dia();i++ ){
					std::auto_ptr<const vector<T,dev_memory_space> > diagonal ( dia.get_dia( dia.get_offset( i ) ));
					dst.set( i , mean( *const_cast<vector<T, dev_memory_space>* >(diagonal.get()) ) );
				}
			}
		template<class T,class I>
			void avg_diagonals(
					cuv::vector<T,host_memory_space,I>& dst,
					const cuv::dia_matrix<T,host_memory_space,I>& dia
					){
				cuvAssert( dia.row_fact( )==1 );
				cuvAssert( dia.num_dia( )==dst.size( ));
				typedef I index_type;
				typedef T value_type;
				unsigned int A_stride = dia.stride();
				unsigned int A_h      = dia.h();
				unsigned int A_w      = dia.w();
				const cuv::vector<int,host_memory_space>& offsets = dia.get_offsets();
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
					dst.set( i , sum/N);
				}
			}

		/***********************************************************
		 * With toeplitz_matrix result
		 ***********************************************************/
		template<class T,class I>
			void avg_diagonals(
					cuv::toeplitz_matrix<T,dev_memory_space,I>& dst,
					const cuv::dia_matrix<T,dev_memory_space,I>& dia
					){
				cuvAssert( false );
			}
		template<class T,class I>
			void avg_diagonals(
					cuv::toeplitz_matrix<T,host_memory_space,I>& dst,
					const cuv::dia_matrix<T,host_memory_space,I>& dia
					){
				cuvAssert( dst.w()==dia.w() );
				cuvAssert( dst.h()==dia.h() );

				typedef I index_type;
				typedef T value_type;

				unsigned int w = dst.w()/dst.output_maps();

				for( int d=0;d<dia.num_dia();d++ ){
					const int off = dia.get_offset( d );  //diagonal offset
					const int virtual_om  = rintf( off/float(w) ); // "output_map" which can be outside matrix (=negative)
					const int virtual_off = off - virtual_om*w;    // the offset relative to the virtual output_map

					const index_type i_start = std::max((int)0,-virtual_off);
					const index_type j_start = std::max((int)0, virtual_off);

					const index_type N = std::min(w - i_start, w - j_start);

					for( int im=0;im<dst.input_maps();im++ ){
						int startx=i_start + im*w;              // x-coordinate of  upper left corner of this diagonal
						if( startx + N > dst.w() )
						   break;
						const T* data = dia.vec().ptr() + d*dia.stride() + startx;
						T avg=std::accumulate(data,data+N,(T)0)/N;
                        /*
						 *for( int i=0;i<N;i++ )
						 *    const_cast<T*>( data )[ i ] = i;
                         */
						dst.vec().set( d*dst.input_maps() + im, avg );
					}
				}
			}
	}

	template<class T, class M, class I>
	void avg_diagonals( cuv::vector<T,M,I>& dst, const cuv::dia_matrix<T,M>& dia ){
		avg_diagonals_impl::avg_diagonals( dst, dia );
	}

	template<class T, class M, class I>
	void avg_diagonals( cuv::toeplitz_matrix<T,M,I>& dst, const cuv::dia_matrix<T,M>& dia ){
		avg_diagonals_impl::avg_diagonals( dst, dia );
	}

	template void
	avg_diagonals(cuv::vector<float,host_memory_space,unsigned int>&, const cuv::dia_matrix<float,host_memory_space>&);
	template void
	avg_diagonals(cuv::vector<float,dev_memory_space,unsigned int>&, const cuv::dia_matrix<float,dev_memory_space>&);

	template void
	avg_diagonals(cuv::toeplitz_matrix<float,host_memory_space,unsigned int>&, const cuv::dia_matrix<float,host_memory_space>&);
	template void
	avg_diagonals(cuv::toeplitz_matrix<float,dev_memory_space,unsigned int>&, const cuv::dia_matrix<float,dev_memory_space>&);
}
