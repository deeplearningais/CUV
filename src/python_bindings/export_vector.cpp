#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>


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

	class_<vec>(name, init<int>())
		.def("size",   &vec::size, "vector size")
		.def("__len__",&vec::size, "vector size")
		.def("memsize",&vec::memsize, "size of vector in memory (bytes)")
		.def("alloc",&vec::alloc, "allocate memory")
		.def("dealloc",&vec::dealloc, "deallocate memory")
		.def("at",  (value_type (vec::*)(int))(&vec::operator[]))
		;
}

void export_vector(){
	export_vector_common<dev_vector<float> >("dev_vector_float");
	export_vector_common<host_vector<float> >("host_vector_float");

	export_vector_common<dev_vector<unsigned char> >("dev_vector_uc");
	export_vector_common<host_vector<unsigned char> >("host_vector_uc");

	def("convert", (void(*)(dev_vector<float>&,const host_vector<float>&)) cuv::convert);
	def("convert", (void(*)(host_vector<float>&,const dev_vector<float>&)) cuv::convert);

	def("convert", (void(*)(dev_vector<unsigned char>&,const host_vector<unsigned char>&)) cuv::convert);
	def("convert", (void(*)(host_vector<unsigned char>&,const dev_vector<unsigned char>&)) cuv::convert);
}

