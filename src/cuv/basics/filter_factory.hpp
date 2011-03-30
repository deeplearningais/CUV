#ifndef __FILTER_FACTORY_HPP__
#define __FILTER_FACTORY_HPP__

#include <cuv/basics/dense_matrix.hpp>
#include <cuv/basics/dia_matrix.hpp>
namespace cuv{
	template<class T, class M, class I=unsigned int>
	class filter_factory{
		public:
			typedef T value_type;
			typedef M memory_space;
			typedef I index_type;
		public:
			filter_factory(int px, int py, int fs, int input_maps, int output_maps)
			: m_px(px)
			, m_py(py)
			, m_fs(fs)
			, m_input_maps(input_maps)
			, m_output_maps(output_maps)
			{
			}

			template<class M2>
			dense_matrix<T,row_major,M,I>*
			extract_filter( const dia_matrix<T,M2>& dia, unsigned int filternumber){
				dense_matrix<T,row_major,M,I>* mat = new dense_matrix<T,row_major,M,I>(m_fs*m_fs, m_input_maps);
				fill(mat->vec(), (T)0);
				unsigned int map_size=dia.h()/m_input_maps;
				for (unsigned int map_num = 0; map_num < m_input_maps; map_num++)
				{
					unsigned int fi = 0;
					for (unsigned int i = 0; i < map_size; ++i) 
					{
						if(!dia.has(i+map_num*map_size,filternumber))
							continue;
						mat->vec().set(map_num * m_fs *m_fs + fi++,dia(i+map_num*map_size,filternumber));
						if(fi>=mat->n())
							break;
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
		private:
			int m_px, m_py, m_fs, m_input_maps, m_output_maps;
			int m_dias_per_filter;
	};
}

#endif  /*__FILTER_FACTORY_HPP__ */
