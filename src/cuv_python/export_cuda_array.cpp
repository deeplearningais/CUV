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
#include <pyublas/numpy.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <cuv/basics/cuda_array.hpp>

using namespace std;
using namespace boost::python;
using namespace cuv;
namespace ublas = boost::numeric::ublas;
namespace bp    = boost::python;

/*
 * Export the cuda_array class
 */
template<class T>
void
export_cuda_array(std::string name){
	typedef T mat;
	typedef typename mat::value_type value_type;
	typedef typename mat::index_type index_type;

	class_<mat>(name.c_str(), init<typename mat::index_type, typename mat::index_type, const unsigned int>())
		.def("__len__",&mat::n, "matrix number of elements")
		.def("alloc",  &mat::alloc, "allocate memory")
		.def("dealloc",&mat::dealloc, "deallocate memory")
		//.def("bind",&mat::bind, "bind to 2D texture")
		//.def("unbind",&mat::unbind, "unbind from 2D texture")
		.def("assign", (void (mat::*)(const tensor<value_type,dev_memory_space,row_major>&))(&mat::assign), "assign a device tensor to cuda_array")
		.def("assign", (void (mat::*)(const tensor<value_type,host_memory_space,row_major>&))(&mat::assign), "assign a host tensor to cuda_array")
		.def("at",    (value_type (mat::*)(const index_type&,const index_type&))(&mat::operator()), "value at this position")
		.add_property("h", &mat::h)
		.add_property("w", &mat::w)
		.add_property("n", &mat::n)
		;
}


/*
 * MAIN export function
 *   calls exporters for various value_types, column/row major combinations etc.
 */
void export_cuda_array(){
	export_cuda_array<cuda_array<float,dev_memory_space> >("dev_cuda_array_f");
	export_cuda_array<cuda_array<unsigned char,dev_memory_space> >("dev_cuda_array_uc");
}


