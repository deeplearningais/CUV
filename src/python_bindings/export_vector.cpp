#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include  <boost/type_traits/is_same.hpp>


#include <dev_vector.hpp>
#include <host_vector.hpp>
#include <convert.hpp>

using namespace boost::python;
using namespace cuv;


template<class T>
void
export_vector_common(const char* name){
	typedef T vec;
	typedef typename vec::value_type value_type;

	class_<vec> (name, init<int>())
		.def("size",   &vec::size, "vector size")
		.def("__len__",&vec::size, "vector size")
		.def("memsize",&vec::memsize, "size of vector in memory (bytes)")
		.def("alloc",&vec::alloc, "allocate memory")
		.def("dealloc",&vec::dealloc, "deallocate memory")
		.def("at",  (value_type  (vec::*)(const typename vec::index_type&)const)(&vec::operator[]))
		;
	
}

template <class T>
void
export_vector_conversion(){
	def("convert", (void(*)(dev_vector<T>&,const host_vector<T>&)) cuv::convert);
	def("convert", (void(*)(host_vector<T>&,const dev_vector<T>&)) cuv::convert);
}

void export_vector(){
	export_vector_common<dev_vector<float> >("dev_vector_float");
	export_vector_common<host_vector<float> >("host_vector_float");

	export_vector_common<dev_vector<unsigned char> >("dev_vector_uc");
	export_vector_common<host_vector<unsigned char> >("host_vector_uc");

	export_vector_conversion<float>();
	export_vector_conversion<unsigned char>();
}

