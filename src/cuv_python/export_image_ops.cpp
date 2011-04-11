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
#include <cuv/basics/cuda_array.hpp>

#include <cuv/image_ops/move.hpp>
#include <cuv/image_ops/image_pyramid.hpp>

//using namespace std;
using namespace boost::python;
using namespace cuv;
namespace ublas = boost::numeric::ublas;

template<class M, class N>
void export_move(){
	def("image_move",
			(void(*)(M&, const N&, const unsigned int&,const unsigned int&,const unsigned int&, const int&, const int&))
			image_move<M,N>, (arg("dst"),arg("src"),arg("image_w"),arg("image_h"),arg("num_maps"),arg("xshift"),arg("yshift")));
}

template<class V, class S, class I>
void export_image_pyramid_functions(){
	def("gaussian",
			(void(*)(tensor<V,S,row_major>&dst, const cuda_array<V,S,I>& src))
			gaussian<V,S,I>, (arg("dst"),arg("src")));
	def("gaussian_pyramid_downsample",
			(void(*)(tensor<V,S,row_major>&dst, const cuda_array<V,S,I>& src, const unsigned int))
			gaussian_pyramid_downsample<V,S,I>, (arg("dst"),arg("src"),arg("interleaved_channels")));
	def("gaussian_pyramid_upsample",
			(void(*)(tensor<V,S,row_major>&dst, const cuda_array<V,S,I>& src))
			gaussian_pyramid_upsample<V,S,I>, (arg("dst"),arg("src")));
}

template<class VDest, class V, class S, class I>
void export_pixel_classes(){
	def("get_pixel_classes",
			(void(*)(tensor<VDest,S,row_major>&dst, 
					 const cuda_array<V,S,I>& src, 
					 float))
			get_pixel_classes<VDest,V,S,I>, (arg("dst"),arg("src"),arg("scale_fact")));
}

template<class M>
void export_image_pyramid(std::string name){
	typedef image_pyramid<M> pyr;
	class_<pyr>(name.c_str(), init<int,int,int,int>())
		.def("get",             &pyr::get,(arg("depth"),arg("channel")=0), return_value_policy<manage_new_object>())
		.def("get_all_channels",  &pyr::get_all_channels,(arg("depth")),   return_internal_reference<>())
		.def("build",           (void (pyr::*)(const M&, const unsigned int)) &pyr::build, (arg("src"), arg("interleaved_channels")=1))
		.add_property("base_h", &pyr::base_h)
		.add_property("base_w", &pyr::base_w)
		.add_property("depth",  &pyr::depth)
		.add_property("dim",    &pyr::dim)
		;
}

void export_image_ops(){
	export_move<dense_matrix<float,dev_memory_space,column_major>,dense_matrix<unsigned char,dev_memory_space,column_major> >();
	export_move<dense_matrix<unsigned char,dev_memory_space,column_major>,dense_matrix<unsigned char,dev_memory_space,column_major> >();
	export_image_pyramid_functions<float,dev_memory_space,unsigned int>();
	export_image_pyramid_functions<unsigned char,dev_memory_space,unsigned int>();
	
	export_pixel_classes<unsigned char, unsigned char,dev_memory_space,unsigned int>();
	export_pixel_classes<unsigned char, float,dev_memory_space,unsigned int>();
	export_pixel_classes<float, float,  dev_memory_space,unsigned int>();

	export_image_pyramid<dense_matrix<float, dev_memory_space,row_major, unsigned int> >("dev_image_pyramid_f");
	export_image_pyramid<dense_matrix<unsigned char, dev_memory_space,row_major, unsigned int> >("dev_image_pyramid_uc");
}
