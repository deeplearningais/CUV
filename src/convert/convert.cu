#include <convert.hpp>

namespace cuv{
template<class Dst, class Src>
void convert(Dst& dst, const Src& src)
{
	convert_impl<Dst,Src>::convert(dst,src);
};

template<class Dst, class Src>
struct convert_impl{

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
		cuvSafeCall(cudaMemcpy(dst.ptr(),src.ptr(),dst.memsize(),cudaMemcpyHostToDevice));
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
		cuvSafeCall(cudaMemcpy(dst.ptr(),src.ptr(),dst.memsize(),cudaMemcpyDeviceToHost));
	}


};





namespace{
	struct instantiator{
		instantiator(){
			dev_dense_matrix<float,column_major> dfc(32,16);
			host_dense_matrix<float,row_major>  hfr(16,32);
			dev_dense_matrix<float,row_major> dfr(32,16);
			host_dense_matrix<float,column_major>  hfc(16,32);
			convert(dfc, hfr);
			convert(hfr, dfc);
		}
	} inst;
}



} // namespace cuv
