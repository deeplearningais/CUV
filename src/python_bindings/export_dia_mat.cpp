#include <string>
#include <iostream>
#include <fstream>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/list.hpp>

#include  <boost/type_traits/is_same.hpp> 

#include <pyublas/numpy.hpp>

#include <host_dia_matrix.hpp>
#include <dev_dia_matrix.hpp>
#include <sparse_matrix_io.hpp>
#include <matrix_ops/matrix_ops.hpp>
#include <matrix_ops/densedense_to_sparse.hpp>
#include <convert.hpp>

namespace ublas = boost::numeric::ublas;
using namespace boost::python;
using namespace cuv;

template<class T>
boost::shared_ptr<T>
create_dia_mat(unsigned int h, unsigned int w, boost::python::list& dia_offsets, unsigned int stride, unsigned int rf=1){
	int num_dia=boost::python::len(dia_offsets);
	int *dias = new int[num_dia];
	for(int i=0; i< num_dia; i++) {
		int ls = boost::python::extract<int>(dia_offsets[i]);
		dias[i]=ls;
	}
	boost::shared_ptr<T> m ( new T(h,w,num_dia,stride,rf));
	m->set_offsets(dias,dias+num_dia);
	delete[] dias;
	return m;
}

template<class T>
boost::shared_ptr<T>
create_dia_mat_empty(){
	boost::shared_ptr<T> m(new T());
	return m;
}
template<class T>
boost::shared_ptr<T>
create_dia_mat_from_dia_mat(T* other){
	int *dias = new int[other->num_dia()];
	for(int i=0; i< other->num_dia(); i++) {
		dias[i]=other->get_offset(i);
	}
	boost::shared_ptr<T> m(new T(other->h(),other->w(),other->num_dia(),other->stride(),other->row_fact()));
	m->set_offsets(dias,dias+other->num_dia());
	delete[] dias;
	return m;
}
template<class V, class I>
struct dia_io{
	static void save_dia_mat(host_dia_matrix<V,I>& m, std::string fn){
			std::ofstream ofs(fn.c_str());
			boost::archive::binary_oarchive oa(ofs);
			oa << m;
	}
	static void load_dia_mat(host_dia_matrix<V,I>& m, std::string fn){
			std::ifstream ifs(fn.c_str());
			boost::archive::binary_iarchive ia(ifs);
			ia >> m;
	}
	static void save_dia_mat(dev_dia_matrix<V,I>& m, std::string fn){
		host_dia_matrix<V,I> m2(m.h(),m.w(),m.num_dia(),m.stride(),m.row_fact());
		convert(m2,m);
		save_dia_mat(m2,fn);
	}
	static void load_dia_mat(dev_dia_matrix<V,I>& m, std::string fn){
		host_dia_matrix<V,I> m2;
		load_dia_mat(m2,fn);
		convert(m,m2);
	}
};

template<class T>
void
export_diamat_common(const char* name){
	typedef T mat;
	typedef typename mat::value_type value_type;
	typedef typename mat::index_type index_type;
	typedef typename mat::vec_type vec_type;

	class_<mat,boost::shared_ptr<mat> > matobj(name);
	matobj
		//.def("w",   &mat::w,    "width")
		//.def("h",   &mat::h,    "height")
		//.def("vec",    (vec_type* (mat::*)())(&mat::vec_ptr), "internal memory vector", return_internal_reference<>())
		.add_property("h", &mat::h)
		.add_property("w", &mat::w)
		.add_property("vec", make_function((vec_type* (mat::*)())(&mat::vec_ptr), return_internal_reference<>()))
		.add_property("stride",&mat::stride, "matrix stride")
		.add_property("num_dia",&mat::num_dia, "number of diagonals")
		.def("__len__",&mat::n, "number of elements")
		.def("alloc",&mat::alloc, "allocate memory")
		.def("dia",&mat::get_dia, "return a view on one of the diagonals", return_value_policy<manage_new_object>())
		.def("dealloc",&mat::dealloc, "deallocate memory")
		.def("save", (void (*)(mat&,std::string)) dia_io<value_type, index_type>::save_dia_mat, "save to file")
		.def("load", (void (*)(mat&,std::string)) dia_io<value_type, index_type>::load_dia_mat, "load from file")
		.def("__call__",  (const value_type& (mat::*)(const typename mat::index_type&, const typename mat::index_type&)const)(&mat::operator()), return_value_policy<copy_const_reference>()) // igitt.
		.def("__init__",  make_constructor(create_dia_mat<mat>))
		.def("__init__",  make_constructor(create_dia_mat_empty<mat>))
		.def("__init__",  make_constructor(create_dia_mat_from_dia_mat<mat>) )
		;


	//def((std::string("make_")+name).c_str(),  create_dia_mat<mat>,              (arg("h"),arg("w"),arg("offsets"),arg("stride"),arg("steepness")=1), return_value_policy<manage_new_object>());
	//def((std::string("make_")+name).c_str(),  create_dia_mat_from_dia_mat<mat>, return_value_policy<manage_new_object>());
}

template<class T>
void export_block_descriptors(const char*name){
	typedef host_block_descriptor<T> hbd;
	class_<hbd>(
			(std::string("host_block_descriptor_") + name).c_str(), init<const host_dia_matrix<T>& >());
	class_<dev_block_descriptor<T>  >(
			(std::string("dev_block_descriptor_") + name).c_str(),  init<const dev_dia_matrix<T>&  >())
		.def("__len__", &dev_block_descriptor<T>::len)
		;
}

// forward declaration...
template<class T, class Mfrom, class Mto_ublas, class Mto_cuv>
pyublas::numpy_matrix<T,Mto_ublas>
host_dense_mat2numpy(dense_matrix<T, Mfrom,host_memory_space>& m);

template<class T>
pyublas::numpy_matrix<T,ublas::column_major> 
dev_dia_mat2numpy(dev_dia_matrix<T>&m){
	host_dia_matrix<T>   hostdia(m.h(),m.w(),m.num_dia(),m.stride(),m.row_fact());
	cuv::convert(hostdia,m);
	dense_matrix<T,column_major,host_memory_space> mdense(m.h(),m.w());
	cuv::convert(mdense,hostdia);
	pyublas::numpy_matrix<T,ublas::column_major> to = host_dense_mat2numpy<T,cuv::column_major,ublas::column_major,cuv::column_major>(mdense);
	return to;
}



template <class T>
void
export_diamat_conversion(){
	def("convert", (void(*)(dev_dia_matrix<T>&,const host_dia_matrix<T>&)) cuv::convert);
	def("convert", (void(*)(host_dia_matrix<T>&,const dev_dia_matrix<T>&)) cuv::convert);
	def("convert", (void(*)(dense_matrix<T,column_major,host_memory_space>&, const host_dia_matrix<T>&)) cuv::convert);
	def("pull",    dev_dia_mat2numpy<T>);
}

void export_dia_matrix(){
	export_diamat_common<dev_dia_matrix<float> >("dev_dia_matrix_f");
	export_diamat_common<host_dia_matrix<float> >("host_dia_matrix_f");
	export_block_descriptors<float>("f");
	export_diamat_conversion<float>();

	//def("densedense_to_dia", densedense_to_dia<dev_dia_matrix<float>, dev_block_descriptor<float>, dev_dense_matrix<float,column_major> >, "C <- A*B', where C is sparse");
	//def("densedense_to_dia", densedense_to_dia<host_dia_matrix<float>,host_block_descriptor<float>,host_dense_matrix<float,column_major> >, "C <- A*B', where C is sparse");

	def("densedense_to_dia", 
			densedense_to_dia<dev_dia_matrix<float>, dev_block_descriptor<float>, dense_matrix<float,column_major,dev_memory_space> >,
			(arg("C"),arg("Cbd"),arg("A"),arg("B"),arg("factAB")=1.f,arg("factC")=0.f));
			//"C <- A*B', where C is sparse");
	def("densedense_to_dia", 
			densedense_to_dia<host_dia_matrix<float>,host_block_descriptor<float>,dense_matrix<float,column_major,host_memory_space> >, 
			(arg("C"),arg("Cbd"),arg("A"),arg("B"),arg("factAB")=1.f,arg("factC")=0.f));
			//"C <- A*B', where C is sparse");

	def("prod", cuv::prod<dense_matrix<float,column_major,host_memory_space>, host_dia_matrix<float>,dense_matrix<float,column_major,host_memory_space> >, 
			(arg("C"),arg("A"),arg("B"),arg("transA"),arg("transB"),arg("factAB")=1.f,arg("factC")=0.f));
	def("prod", cuv::prod<dense_matrix<float,column_major,dev_memory_space>,  dev_dia_matrix<float>,  dense_matrix<float,column_major,dev_memory_space> >,
			(arg("C"),arg("A"),arg("B"),arg("transA"),arg("transB"),arg("factAB")=1.f,arg("factC")=0.f));
}
