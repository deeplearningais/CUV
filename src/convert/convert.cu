#include <convert.hpp>

namespace cuv{

	template<class Dst,class Src>
		struct convert_impl;

	template<class Dst, class Src>
		void convert(Dst& dst, const Src& src)
		{
			convert_impl<Dst,Src>::convert(dst,src);
		};

	template<class Dst, class Src>
		struct convert_impl{

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
					convert(*dst.vec(), *src.vec());
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
					convert(*dst.vec(), *src.vec());
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
					convert(*dst.vec(), *src.vec());
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
					convert(*dst.vec(), *src.vec());
				}


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
		};

	// Matrix Conversion
	template void convert<dev_dense_matrix<float,column_major>,         host_dense_matrix<float,row_major> >
		(                 dev_dense_matrix<float,column_major>&,  const host_dense_matrix<float,row_major>&);
	template void convert<dev_dense_matrix<float,row_major>,            host_dense_matrix<float,column_major> >
		(                 dev_dense_matrix<float,row_major>&,     const host_dense_matrix<float,column_major>&);
	template void convert<host_dense_matrix<float,column_major>,        dev_dense_matrix<float,row_major> >
		(                 host_dense_matrix<float,column_major>&, const dev_dense_matrix<float,row_major>&);
	template void convert<host_dense_matrix<float,row_major>,           dev_dense_matrix<float,column_major> >
		(                 host_dense_matrix<float,row_major>&,    const dev_dense_matrix<float,column_major>&);

	// Vector Conversion (implicitly instantiated by matrix conversions already, included for completeness)
	template void convert<host_vector<float>,           dev_vector<float> >
		(                 host_vector<float>&,    const dev_vector<float>&);
	template void convert<dev_vector<float>,           host_vector<float> >
		(                 dev_vector<float>&,    const host_vector<float>&);


} // namespace cuv


