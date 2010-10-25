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


#include <cuv_general.hpp>
#include <random/random.hpp>

using namespace boost::python;
using namespace cuv;

void export_vector();
void export_vector_ops();
void export_dense_matrix();
void export_cuda_array();
void export_matrix_ops();
void export_random();
void export_dia_matrix();
void export_convolution_ops();
void export_image_ops();
void export_tools();
void export_libs_rbm();

BOOST_PYTHON_MODULE(_cuv_python){
	def("initCUDA", initCUDA);
	def("exitCUDA", exitCUDA);
	def("safeThreadSync", safeThreadSync);
	def("initialize_mersenne_twister_seeds", initialize_mersenne_twister_seeds);
	export_vector();
	export_vector_ops();
	export_dense_matrix();
	export_cuda_array();
	export_matrix_ops();
	export_random();
	export_dia_matrix();
	export_convolution_ops();
	export_image_ops();
	export_tools();
	export_libs_rbm();
}


