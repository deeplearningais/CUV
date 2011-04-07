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

#include <cuv/basics/dense_matrix.hpp>
#include <cuv/libs/rbm/rbm.hpp>

//using namespace std;
using namespace boost::python;
using namespace cuv;
using namespace cuv::libs::rbm;
namespace ublas = boost::numeric::ublas;

template<class V, class M, class L, class I>
void export_libs_rbm_detail(){
	typedef dense_matrix<V,M,L,I> mat;
	typedef tensor<V,M> vec;
	def("set_binary_sequence", set_binary_sequence<mat>, (arg("matrix"), arg("startvalue")));
	def("sigm_temperature", sigm_temperature<mat,vec>, (arg("matrix"), arg("temperature")));
}

template<class V, class M, class L, class I>
void export_set_local_conn(){
	typedef dense_matrix<V,M,L,I> mat;
	typedef tensor<V,M> vec;
	def("set_local_connectivity_in_dense_matrix", set_local_connectivity_in_dense_matrix<mat>, (arg("matrix"),arg("patchsize"),arg("px"),arg("py"),arg("pxh"),arg("pyh"),arg("maxdist_from_main_dia"),arg("round")=false));
}

template<class V, class M, class L, class I>
void export_copy_at_rowidx(){
	typedef dense_matrix<V,M,L,I> mat;
	typedef tensor<V,M> vec;
	def("copy_at_rowidx", copy_at_rowidx<mat,mat>, (arg("dst"), arg("src"),arg("rowidx"),arg("offset")));
	def("copy_redblack", copy_redblack<mat>, (arg("dst"), arg("src"),arg("num_maps"), arg("color")));
}

void export_libs_rbm(){
	export_libs_rbm_detail<float,host_memory_space,column_major,unsigned int>();
	export_libs_rbm_detail<float,dev_memory_space,column_major,unsigned int>();
	export_set_local_conn<float,dev_memory_space,column_major,unsigned int>();
	export_copy_at_rowidx<float,dev_memory_space,column_major,unsigned int>();
}
