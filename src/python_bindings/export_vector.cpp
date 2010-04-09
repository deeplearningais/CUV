#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include  <boost/type_traits/is_same.hpp>


#include <vector.hpp>
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
		.def("set",    &vec::set, "set index to value")
		.def("at",  (value_type  (vec::*)(const typename vec::index_type&)const)(&vec::operator[]))
		;
	
}

template <class T>
void
export_vector_conversion(){
	def("convert", (void(*)(vector<T,dev_memory_space>&,const vector<T,host_memory_space>&)) cuv::convert);
	def("convert", (void(*)(vector<T,host_memory_space>&,const vector<T,dev_memory_space>&)) cuv::convert);
}

void export_vector(){
	export_vector_common<vector<float,dev_memory_space> >("dev_vector_float");
	export_vector_common<vector<float,host_memory_space> >("host_vector_float");

	export_vector_common<vector<unsigned char,dev_memory_space> >("dev_vector_uc");
	export_vector_common<vector<unsigned char,host_memory_space> >("host_vector_uc");

	export_vector_common<vector<int,dev_memory_space> >("dev_vector_int");
	export_vector_common<vector<int,host_memory_space> >("host_vector_int");

	export_vector_conversion<float>();
	export_vector_conversion<unsigned char>();
	export_vector_conversion<int>();
}

