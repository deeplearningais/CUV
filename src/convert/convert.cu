#include <dev_dia_matrix.hpp>
#include <host_dia_matrix.hpp>
#include <vector_ops/vector_ops.hpp>
#include <convert.hpp>

namespace cuv{

		namespace convert_impl{
			/*
			 * Vector Conversion
			 */


			// host  --> dev 
			template<class __value_type, class __index_type>
				static void
				convert(        dev_vector <__value_type,   __index_type>& dst, 
						const host_vector<__value_type,  __index_type>& src){
					if(        dst.size() != src.size()){
						dev_vector<__value_type,__index_type> d(src.size());
						dst = d;
					}
					cuvSafeCall(cudaMemcpy(dst.ptr(),src.ptr(),dst.memsize(),cudaMemcpyHostToDevice));
				}

			// dev  --> host  
			template<class __value_type, class __index_type>
				static void
				convert(        host_vector<__value_type,  __index_type>& dst, 
						const dev_vector<__value_type,  __index_type>& src){
					if( dst.size() != src.size()){
						host_vector<__value_type,__index_type> h(src.size());
						dst = h;
					}
					cuvSafeCall(cudaMemcpy(dst.ptr(),src.ptr(),dst.memsize(),cudaMemcpyDeviceToHost));
				}

			/*
			 * Matrix Conversion
			 */

			// host (row-major) --> dev (col-major) 
			template<class __value_type, class __index_type>
				static void
				convert(        dev_dense_matrix <__value_type,  column_major,  __index_type>& dst, 
						const host_dense_matrix<__value_type,  row_major, __index_type>& src){
					if(        dst.h() != src.w()
							|| dst.w() != src.h()){

						dev_dense_matrix<__value_type,column_major,__index_type> d(src.w(),src.h());
						dst = d;
					}
					convert(dst.vec(), src.vec());
				}

			// dev (col-major) --> host (row-major) 
			template<class __value_type, class __index_type>
				static void
				convert(        host_dense_matrix<__value_type,  row_major,  __index_type>& dst, 
						const dev_dense_matrix<__value_type,  column_major, __index_type>& src){
					if(        dst.h() != src.w()
							|| dst.w() != src.h()){
						host_dense_matrix<__value_type,row_major,__index_type> h(src.w(),src.h());
						dst = h;
					}
					convert(dst.vec(), src.vec());
				}

			// host (col-major) --> dev (row-major) 
			template<class __value_type, class __index_type>
				static void
				convert(        dev_dense_matrix <__value_type,  row_major,  __index_type>& dst, 
						const host_dense_matrix<__value_type,  column_major, __index_type>& src){
					if(        dst.h() != src.w()
							|| dst.w() != src.h()){

						dev_dense_matrix<__value_type,row_major,__index_type> d(src.w(),src.h());
						dst = d;
					}
					convert(dst.vec(), src.vec());
				}

			// dev (row-major) --> host (col-major) 
			template<class __value_type, class __index_type>
				static void
				convert(        host_dense_matrix<__value_type,  column_major,  __index_type>& dst, 
						const dev_dense_matrix<__value_type,  row_major, __index_type>& src){
					if(        dst.h() != src.w()
							|| dst.w() != src.h()){
						host_dense_matrix<__value_type,column_major,__index_type> h(src.w(),src.h());
						dst = h;
					}
					convert(dst.vec(), src.vec());
				}

			/*
			 * Simple copying
			 *
			 */

			// dev (col-major) --> host (col-major) 
			template<class __value_type, class __index_type>
				static void
				convert(        host_dense_matrix<__value_type,  column_major, __index_type>& dst, 
						const    dev_dense_matrix<__value_type,  column_major, __index_type>& src){
					if(        dst.h() != src.h()
							|| dst.w() != src.w()){
						host_dense_matrix<__value_type,column_major,__index_type> h(src.h(),src.w());
						dst = h;
					}
					convert(dst.vec(), src.vec());
				}

			// dev (col-major) --> host (col-major) 
			template<class __value_type, class __index_type>
				static void
				convert(         dev_dense_matrix<__value_type,  column_major, __index_type>& dst, 
						const   host_dense_matrix<__value_type,  column_major, __index_type>& src){
					if(        dst.h() != src.h()
							|| dst.w() != src.w()){
						dev_dense_matrix<__value_type,column_major,__index_type> h(src.h(),src.w());
						dst = h;
					}
					convert(dst.vec(), src.vec());
				}

			// dev (row-major) --> host (row-major) 
			template<class __value_type, class __index_type>
				static void
				convert(        host_dense_matrix<__value_type,  row_major,  __index_type>& dst, 
						const dev_dense_matrix<__value_type,  row_major, __index_type>& src){
					if(        dst.h() != src.h()
							|| dst.w() != src.w()){
						host_dense_matrix<__value_type,row_major,__index_type> h(src.h(),src.w());
						dst = h;
					}
					convert(dst.vec(), src.vec());
				}

			// host (row-major) --> dev (row-major) 
			template<class __value_type, class __index_type>
				static void
				convert(         dev_dense_matrix<__value_type,  row_major,  __index_type>& dst, 
						const host_dense_matrix<__value_type,  row_major, __index_type>& src){
					if(        dst.h() != src.h()
							|| dst.w() != src.w()){
						dev_dense_matrix<__value_type,row_major,__index_type> h(src.h(),src.w());
						dst = h;
					}
					convert(dst.vec(), src.vec());
				}

			/*
			 * Host Dia -> Host Dense
			 */
			template<class __value_type, class __mem_layout_type, class __index_type>
				static void
				convert(      host_dense_matrix <__value_type,   __mem_layout_type, __index_type>& dst, 
						const host_dia_matrix<__value_type,  __index_type>& src){
					if(        dst.h() != src.h()
							|| dst.w() != src.w()
							){
						host_dense_matrix<__value_type,__mem_layout_type,__index_type> d(src.h(),src.w());
						dst = d;
					}
					fill(dst.vec(),0);
					const host_vector<int>& off = src.get_offsets();
					using namespace std;
					const int rf = src.row_fact();
					for(unsigned int oi=0; oi < off.size(); oi++){
						int o = off[oi];
						__index_type j = 1 *max((int)0, o);
						__index_type i = rf*max((int)0,-o);
						for(;i<src.h() && j<src.w(); j++){
							for(int k=0;k<rf;k++,i++)
								dst.set(i,j, src(i,j));
						}
					}
				}

			/*
			 * Host Dia -> Dev Dia
			 */
			template<class __value_type, class __index_type>
				static void
				convert(      dev_dia_matrix <__value_type, __index_type>& dst, 
						const host_dia_matrix<__value_type, __index_type>& src){
					if(        dst.h() != src.h()
							|| dst.w() != src.w()
							|| dst.row_fact() != src.row_fact()
							|| !dst.vec_ptr()
							){
						dst.dealloc();
						dst = dev_dia_matrix<__value_type,__index_type>(src.h(),src.w(),src.num_dia(),src.stride(),src.row_fact());
					}
					cuv::convert(dst.get_offsets(), src.get_offsets());
					cuv::convert(dst.vec(), src.vec());
					dst.post_update_offsets();
				}

			/*
			 * Dev Dia -> Host Dia
			 */
			template<class __value_type, class __index_type>
				static void
				convert(      host_dia_matrix <__value_type, __index_type>& dst, 
						const dev_dia_matrix<__value_type, __index_type>& src){
					if(        dst.h() != src.h()
							|| dst.w() != src.w()
							|| dst.row_fact() != src.row_fact()
							|| !dst.vec_ptr()
							){
						dst.dealloc();
						dst = host_dia_matrix<__value_type,__index_type>(src.h(),src.w(),src.num_dia(),src.stride(),src.row_fact());
					}
					cuv::convert(dst.get_offsets(), src.get_offsets());
					cuv::convert(dst.vec(), src.vec());
					dst.post_update_offsets();
				}


		};
	template<class Dst, class Src>
		void convert(Dst& dst, const Src& src)
		{
			convert_impl::convert<typename Dst::value_type, typename Dst::index_type>(dst,src); // hmm the compiler should deduce template args, but it fails to do so.
		};

#define CONV_VEC(X) \
	template void convert<dev_vector<X>,          host_vector<X> > \
		(                 dev_vector<X>&,   const host_vector<X>&); \
	template void convert<host_vector<X>,          dev_vector<X> > \
		(                 host_vector<X>&,   const dev_vector<X>&);

#define CONV_INST(X,Y,Z) \
	template void convert<dev_dense_matrix<X,Y>,          host_dense_matrix<X,Z> > \
		(                 dev_dense_matrix<X,Y>&,   const host_dense_matrix<X,Z>&); \
	template void convert<host_dense_matrix<X,Y>,         dev_dense_matrix<X,Z> > \
		(                 host_dense_matrix<X,Y>&,  const dev_dense_matrix<X,Z>&);

CONV_INST(float,column_major,column_major);
CONV_INST(float,column_major,row_major);
CONV_INST(float,row_major,   column_major);
CONV_INST(float,row_major,   row_major);

CONV_INST(unsigned char,column_major,column_major);
CONV_INST(unsigned char,column_major,row_major);
CONV_INST(unsigned char,row_major,   column_major);
CONV_INST(unsigned char,row_major,   row_major);

CONV_INST(signed char,column_major,column_major);
CONV_INST(signed char,column_major,row_major);
CONV_INST(signed char,row_major,   column_major);
CONV_INST(signed char,row_major,   row_major);

CONV_VEC(float);
CONV_VEC(int);
CONV_VEC(unsigned char);
CONV_VEC(signed char);

#define DIA_DENSE_CONV(X,Y,Z) \
	template <>                           \
		void convert(host_dense_matrix<X,Y,Z>& dst, const host_dia_matrix<X,Z>& src)     \
		{                                                                                \
			typedef host_dense_matrix<X,Y,Z> Dst;                                        \
			convert_impl::convert<typename Dst::value_type, typename Dst::memory_layout, typename Dst::index_type>(dst,src);  \
		};   
#define DIA_HOST_DEV_CONV(X,Z) \
	template <>                           \
		void convert(dev_dia_matrix<X,Z>& dst, const host_dia_matrix<X,Z>& src)     \
		{                                                                                \
			typedef dev_dia_matrix<X,Z> Dst;                                        \
			convert_impl::convert<typename Dst::value_type, typename Dst::index_type>(dst,src);  \
		};                                \
	template <>                           \
		void convert(host_dia_matrix<X,Z>& dst, const dev_dia_matrix<X,Z>& src)     \
		{                                                                                \
			typedef host_dia_matrix<X,Z> Dst;                                        \
			convert_impl::convert<typename Dst::value_type, typename Dst::index_type>(dst,src);  \
		}; 
        
DIA_DENSE_CONV(float,column_major,unsigned int)
DIA_DENSE_CONV(float,row_major,unsigned int)
DIA_HOST_DEV_CONV(float,unsigned int)


} // namespace cuv


