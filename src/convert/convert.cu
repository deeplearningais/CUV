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

			// dev (row-major) --> host (row-major) 
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

CONV_VEC(float);
CONV_VEC(unsigned char);


} // namespace cuv


