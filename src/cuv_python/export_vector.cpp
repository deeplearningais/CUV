//*LB*
// Copyright (c) 2010, University of Bonn, Institute for Computer Science VI
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of the University of Bonn 
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*LE*





#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include  <boost/type_traits/is_same.hpp>


#include <vector.hpp>
#include <convert.hpp>

using namespace boost::python;
using namespace cuv;

template<class T>
long int this_ptr(const T& t){
	return (long int)(&t);
}
template<class T>
long int internal_ptr(const T& t){
	return (long int)(t.ptr());
}

template<class T>
void
export_vector_common(const char* name){
	typedef T vec;
	typedef typename vec::value_type value_type;

	class_<vec> (name, init<int>())
		.def("__len__",&vec::size, "vector size")
		.def("alloc",&vec::alloc, "allocate memory")
		.def("dealloc",&vec::dealloc, "deallocate memory")
		.def("set",    &vec::set, "set index to value")
		.def("at",  (value_type  (vec::*)(const typename vec::index_type&)const)(&vec::operator[]))
		.add_property("size", &vec::size)
		.add_property("memsize",&vec::memsize, "size of vector in memory (bytes)")
		;
	def("this_ptr", this_ptr<vec>);
	def("internal_ptr", internal_ptr<vec>);
	
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

	export_vector_common<vector<unsigned int,dev_memory_space> >("dev_vector_uint");
	export_vector_common<vector<unsigned int,host_memory_space> >("host_vector_uint");

	export_vector_conversion<float>();
	export_vector_conversion<unsigned char>();
	export_vector_conversion<int>();
	export_vector_conversion<unsigned int>();
	}

