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

#include <cuv/basics/tensor.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/random/random.hpp>

//using namespace std;
using namespace boost::python;
using namespace cuv;
namespace ublas = boost::numeric::ublas;

template <class T>
void export_functions() {
	def("add_rnd_normal",add_rnd_normal<typename T::value_type, typename T::memory_space_type, typename T::memory_layout_type>,(arg("dst"),arg("std")=1));
	def("fill_rnd_uniform",fill_rnd_uniform<typename T::value_type, typename T::memory_space_type, typename T::memory_layout_type>,(arg("dst")));
	def("rnd_binarize",rnd_binarize<typename T::value_type, typename T::memory_space_type, typename T::memory_layout_type>,(arg("dst")));
}

void export_random(){
	typedef tensor<float,dev_memory_space,column_major> fdev_cm;
	typedef tensor<float,host_memory_space,column_major> fhost_cm;
	typedef tensor<float,dev_memory_space> fdev;
	typedef tensor<float,host_memory_space> fhost;
	export_functions<fdev>();
	export_functions<fhost>();
	export_functions<fdev_cm>();
	export_functions<fhost_cm>();
	}
