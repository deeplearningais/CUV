#ifndef __FILTER_FACTORY_HPP__
#define __FILTER_FACTORY_HPP__

#include <basics/dense_matrix.hpp>
#include <basics/toeplitz_matrix.hpp>
#include <basics/dia_matrix.hpp>
namespace cuv{
	template<class T, class M, class I=unsigned int>
	class filter_factory{
		public:
			filter_factory(int px, int py, int fs, int input_maps, int output_maps)
			: m_px(px)
			, m_py(py)
			, m_fs(fs)
			, m_input_maps(input_maps)
			, m_output_maps(output_maps)
			{
			}

			toeplitz_matrix<T,M>* 
			create_toeplitz_from_filters(const dense_matrix<T,column_major,M,I>& mat){
				toeplitz_matrix<T,M>* tp_ptr = get_toeplitz();
				toeplitz_matrix<T,M>& tp     = *tp_ptr;
				fill( tp.vec(), 0.f );


				int w   = m_px*m_py;
				for( unsigned int dia = 0; dia < tp.num_dia(); dia++){
					// determine the input/output map that this element belongs to (or whether at all)
					// we do this by calculating the coordinate of the
					// central filter pixel and checking whether it is
					// inside the matrix.
					const int central_dia = ( dia/m_dias_per_filter )*m_dias_per_filter + ( m_fs*m_fs )/2;
					const int off = tp.get_offset( central_dia );
					for( int f=0; f<m_input_maps;f++ ){
						// this number is unique for each diagonal block in toeplitz matrix
						//int filter_num = (dia / m_dias_per_filter) * m_input_maps + f;


						// coordinate of the top left corner of the diagonal
						// move along the diagonal to the start of the actual filter block
						const int tpx=off+f*w,  tpy=f*w;

						if( tpx < 0       || tpx >= tp.w()) {
							 continue;
						 }

						// determine input and output map for the target coordinates
						unsigned int input_map  = tpy/w;
						unsigned int output_map = tpx/w;
						
						// and write it to the filter position in the target matrix
						unsigned int fy = ( dia%m_dias_per_filter ) / m_fs;
						unsigned int fx = ( dia%m_dias_per_filter ) % m_fs;

						// now we can simply get the filter value from the toeplitz matrix
						unsigned int tpidx =  dia * m_input_maps + input_map ;
						T val     = ( mat )( ( unsigned int ) (fy*m_fs + fx),  input_map*m_output_maps + output_map);
						tp.vec().set( tpidx, val );
						//tp.vec().set( tpidx, 1 );

					}
				}
				return tp_ptr;
			}

			dense_matrix<T,column_major,M,I>*
			extract_filters( const toeplitz_matrix<T,M>& tp){
				dense_matrix<T,column_major,M,I>* mat = new dense_matrix<T,column_major,M,I>(m_fs*m_fs, m_input_maps*m_output_maps);
				int w   = m_px*m_py;
				for( unsigned int dia = 0; dia < tp.num_dia(); dia++){
					// determine the input/output map that this element belongs to (or whether at all)
					// we do this by calculating the coordinate of the
					// central filter pixel and checking whether it is
					// inside the matrix.
					int central_dia = ( dia/m_dias_per_filter )*m_dias_per_filter + ( m_fs*m_fs )/2;
					int off = tp.get_offset( central_dia );
					for( int f=0; f<m_input_maps;f++ ){
						// coordinate of the top left corner of the diagonal
						// move along the diagonal to the start of the actual filter block
						int tpx=off+f*w,tpy=f*w;
						if( tpx < 0 || tpx >= tp.w()) continue;

						// determine input and output map for the target coordinates
						unsigned int input_map  = tpy/w;
						unsigned int output_map = tpx/w;

						// now we can simply get the filter value from the toeplitz matrix
						T val = tp.vec()[ dia*m_input_maps + input_map ];
						
						// and write it to the filter position in the target matrix
						unsigned int fy = ( dia%m_dias_per_filter ) / m_fs;
						unsigned int fx = ( dia%m_dias_per_filter ) % m_fs;
						mat->set( ( unsigned int ) (fy*m_fs + fx),  input_map*m_output_maps + output_map, val);
					}
				}
				return mat;
			}

			dia_matrix<T,M,I>*
			get_dia(){
				int fs = m_fs;
				int nm = m_output_maps;
				int msize = fs*fs*( nm + m_input_maps-1 );
				int* off = new int[ msize ];
				int offidx=0;
				int dias_per_filter = 0;
				dia_matrix<T,M>* tp = new dia_matrix<T,M>(
						m_px*m_py*m_input_maps, 
						m_px*m_py*m_output_maps,
						msize,
						std::max(m_px*m_py*m_output_maps, 
							m_px*m_py*m_input_maps));
				for( int m=0;m<nm+m_input_maps-1;m++ ){ 
					dias_per_filter = 0;
					for( int i=0;i<fs;i++ ){
						for( int j=0;j<fs;j++ ){ 
							off[ offidx++ ] = i*m_px+j + m*m_px*m_py;
							dias_per_filter ++;
						}
					}
				}
				cuvAssert( offidx == msize );

				m_dias_per_filter = dias_per_filter;

				for( int i=0;i<msize;i++ )
					off[ i ] += -( m_px+1 )*( int( fs/2 ) ) - ( m_input_maps-1 ) * m_px*m_py;
				tp->set_offsets( off, off+msize );

				delete off;
				return tp;
			}
			toeplitz_matrix<T,M,I>*
			get_toeplitz(){
				int fs = m_fs;
				int nm = m_output_maps;
				int msize = fs*fs*( nm + m_input_maps-1 );
				int* off = new int[ msize ];
				int offidx=0;
				int dias_per_filter = 0;

				toeplitz_matrix<T,M>* tp = new toeplitz_matrix<T,M>(
						m_px*m_py*m_input_maps, 
						m_px*m_py*m_output_maps,
						msize,
						m_input_maps,
						m_output_maps);

				for( int m=0;m<nm+m_input_maps-1;m++ ){ 
					dias_per_filter = 0;
					for( int i=0;i<fs;i++ ){
						for( int j=0;j<fs;j++ ){ 
							off[ offidx++ ] = i*m_px+j + m*m_px*m_py;
							dias_per_filter ++;
						}
					}
				}
				std::cout << "Created " << offidx << ", need" << tp->get_offsets().size()<<" diagonals. "<<std::endl;
				cuvAssert( offidx == msize );
				cuvAssert( offidx == tp->get_offsets().size() );

				m_dias_per_filter = dias_per_filter;

				for( int i=0;i<msize;i++ )
					off[ i ] += -( m_px+1 )*( int( fs/2 ) ) - ( m_input_maps-1 ) * m_px*m_py;
				tp->set_offsets( off, off+msize );
				tp->post_update_offsets();

				delete off;
				return tp;
			}


		private:
			int m_px, m_py, m_fs, m_input_maps, m_output_maps;
			int m_dias_per_filter;
	};
}

#endif  /*__FILTER_FACTORY_HPP__ */
