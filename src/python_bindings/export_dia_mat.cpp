#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/list.hpp>


#include <host_dia_matrix.hpp>
#include <dev_dia_matrix.hpp>
#include <matrix_ops/matrix_ops.hpp>
#include <matrix_ops/densedense_to_sparse.hpp>
#include <convert.hpp>

using namespace boost::python;
using namespace cuv;

template<class T>
T*
create_dia_mat(unsigned int h, unsigned int w, boost::python::list& dia_offsets, unsigned int stride){
	int num_dia=boost::python::len(dia_offsets);
	int *dias = new int[num_dia];
	for(int i=0; i< num_dia; i++) {
		int ls = boost::python::extract<int>(dia_offsets[i]);
		dias[i]=ls;
	}
	T* m = new T(h,w,num_dia,stride);
	m->set_offsets(dias,dias+num_dia);
	return m;
}

template<class T>
void
export_diamat_common(const char* name){
	typedef T mat;
	typedef typename mat::value_type value_type;

	class_<mat>(name, no_init)
		.def("w",   &mat::w,    "width")
		.def("h",   &mat::h,    "height")
		.def("__len__",&mat::n, "number of elements")
		.def("alloc",&mat::alloc, "allocate memory")
		.def("dealloc",&mat::dealloc, "deallocate memory")
		.def("__call__",  (const value_type& (mat::*)(const typename mat::index_type&, const typename mat::index_type&)const)(&mat::operator()), return_value_policy<copy_const_reference>()) // igitt.
		;
	def((std::string("make_")+name).c_str(),  create_dia_mat<mat>, return_value_policy<manage_new_object>());
}

template<class T>
void export_block_descriptors(const char*name){
	typedef host_block_descriptor<T> hbd;
	class_<hbd>(
			(std::string("host_block_descriptor_") + name).c_str(), init<const host_dia_matrix<T>& >());
	class_<dev_block_descriptor<T>  >(
			(std::string("dev_block_descriptor_") + name).c_str(),  init<const dev_dia_matrix<T>&  >())
}



template <class T>
void
export_diamat_conversion(){
	def("convert", (void(*)(dev_dia_matrix<T>&,const host_dia_matrix<T>&)) cuv::convert);
	def("convert", (void(*)(host_dia_matrix<T>&,const dev_dia_matrix<T>&)) cuv::convert);
}

void export_dia_matrix(){
	export_diamat_common<dev_dia_matrix<float> >("dev_dia_matrix_f");
	export_diamat_common<host_dia_matrix<float> >("host_dia_matrix_f");
	export_block_descriptors<float>("f");
	export_diamat_conversion<float>();

	//def("densedense_to_dia", densedense_to_dia<dev_dia_matrix<float>, dev_block_descriptor<float>, dev_dense_matrix<float,column_major> >, "C <- A*B', where C is sparse");
	//def("densedense_to_dia", densedense_to_dia<host_dia_matrix<float>,host_block_descriptor<float>,host_dense_matrix<float,column_major> >, "C <- A*B', where C is sparse");

	def("densedense_to_dia", 
			densedense_to_dia<dev_dia_matrix<float>, dev_block_descriptor<float>, dev_dense_matrix<float,column_major> >,
			(arg("C"),arg("Cbd"),arg("A"),arg("B"),arg("factAB")=1.f,arg("factC")=0.f));
			//"C <- A*B', where C is sparse");
	def("densedense_to_dia", 
			densedense_to_dia<host_dia_matrix<float>,host_block_descriptor<float>,host_dense_matrix<float,column_major> >, 
			(arg("C"),arg("Cbd"),arg("A"),arg("B"),arg("factAB")=1.f,arg("factC")=0.f));
			//"C <- A*B', where C is sparse");

	def("prod", cuv::prod<host_dense_matrix<float,column_major>, host_dia_matrix<float>,host_dense_matrix<float,column_major> >, "C <- A*B', where C is sparse");
	def("prod", cuv::prod<dev_dense_matrix<float,column_major>,  dev_dia_matrix<float>,  dev_dense_matrix<float,column_major> >, "C <- A*B', where C is sparse");
}
