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
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>


//using namespace std;
using namespace boost::python;
using namespace cuv;
namespace ublas = boost::numeric::ublas;




template <class T>
void export_convolve(){
    typedef typename T::value_type V;
    typedef typename T::memory_space_type M;
    typedef typename T::memory_layout_type L;
    typedef cuv::tensor<int,M,L> IT;

    using namespace cuv::alex_conv;

    def("reorder_for_conv",(void (*)(T&,const T&))reorder_for_conv<V,M,L>, (
                arg("dst"),
                arg("src")));
    def("reorder_from_conv",(void (*)(T&,const T&))reorder_from_conv<V,M,L>, (
                arg("dst"),
                arg("src")));
    def("convolve2d",(void (*)(T&,const T&,const T&,int, unsigned int, unsigned int, float, float)) convolve2d<V,M,L>, (
                arg("dst"),
                arg("img"),
                arg("filter"),
                arg("padding_start")=0,
                arg("module_stride")=0,
                arg("n_groups")=0,
                arg("fact_new")=1.f,
                arg("fact_old")=0.f
                ));

    def("convolve2d",(void (*)(T&,const T&,const T&,const IT&, int, unsigned int, unsigned int, float, float)) convolve2d<V,M,L>, (
                arg("dst"),
                arg("img"),
                arg("filter"),
                arg("indices"),
                arg("padding_start")=0,
                arg("module_stride")=0,
                arg("n_groups")=0,
                arg("fact_new")=1.f,
                arg("fact_old")=0.f
                ));

    def("d_conv2d_dimg",(void (*)(T&,const T&,const T&,int, unsigned int, unsigned int, float, float)) d_conv2d_dimg<V,M,L>, (
                arg("dst"),
                arg("delta"),
                arg("filter"),
                arg("padding_start")=0,
                arg("module_stride")=0,
                arg("n_groups")=0,
                arg("fact_new")=1.f,
                arg("fact_old")=0.f
                ));

    def("d_conv2d_dimg",(void (*)(T&,const T&,const T&,const IT&,int, unsigned int, unsigned int, float, float)) d_conv2d_dimg<V,M,L>, (
                arg("dst"),
                arg("delta"),
                arg("filter"),
                arg("indices"),
                arg("padding_start")=0,
                arg("module_stride")=0,
                arg("n_groups")=0,
                arg("fact_new")=1.f,
                arg("fact_old")=0.f
                ));

    def("d_conv2d_dfilt",(void (*)(T&,const T&,const T&,int, unsigned int, unsigned int, unsigned int, float, float)) d_conv2d_dfilt<V,M,L>, (
                arg("dst"),
                arg("delta"),
                arg("input"),
                arg("padding_start")=0,
                arg("module_stride")=0,
                arg("n_groups")=0,
                arg("partial_sum")=1,
                arg("fact_new")=1.f,
                arg("fact_old")=0.f
                ));
    def("d_conv2d_dfilt",(void (*)(T&,const T&,const T&, const IT&, int, unsigned int, unsigned int, unsigned int, float, float)) d_conv2d_dfilt<V,M,L>, (
                arg("dst"),
                arg("delta"),
                arg("input"),
                arg("indices"),
                arg("padding_start")=0,
                arg("module_stride")=0,
                arg("n_groups")=0,
                arg("partial_sum")=1,
                arg("fact_new")=1.f,
                arg("fact_old")=0.f
                ));


    def("local_pool",(void (*)(T&,const T&, int, int, int, int, pool_type)) local_pool<V,M,L>, (
                arg("dst"),
                arg("images"),
                arg("subsx"),
                arg("startx"),
                arg("stridex"),
                arg("outputsx"),
                arg("pool_type")));

    def("local_max_pool_grad",(void (*)(T&,const T&, const T&, const T&, int, int, int, float, float)) local_max_pool_grad<V,M,L>, (
                arg("target"),
                arg("images"),
                arg("maxGrads"),
                arg("maxActs"),
                arg("subsx"),
                arg("startx"),
                arg("stridex"),
                arg("fact_new")=1.f,
                arg("fact_old")=0.f));

    def("local_avg_pool_grad",(void (*)(T&,const T&, int, int, int)) local_avg_pool_grad<V,M,L>, (
                arg("target"),
                arg("avgGrads"),
                arg("subsx"),
                arg("startx"),
                arg("stridex")));

    def("response_normalization",(void (*)(T&, T&, const T&, int, float, float)) response_normalization<V,M,L>, (
                arg("target"),
                arg("denoms"),
                arg("images"),
                arg("patch_size"),
                arg("add_scale"),
                arg("pow_scale")));

    def("response_normalization_grad",(void (*)(T&, T&, const T&, const T&, const T&, int, float, float, float, float)) response_normalization_grad<V,M,L>, (
                arg("input_gradients"),
                arg("original_outputs"),
                arg("original_inputs"),
                arg("delta"),
                arg("denoms"),
                arg("patch_size"),
                arg("add_scale"),
                arg("pow_scale"),
                arg("fact_new")=1.f,
                arg("fact_old")=0.f
                ));

    def("contrast_normalization",(void (*)(T&, T&, const T&, const T&, int, float, float)) contrast_normalization<V,M,L>, (
                arg("target"),
                arg("denoms"),
                arg("mean_diffs"),
                arg("images"),
                arg("patch_size"),
                arg("add_scale"),
                arg("pow_scale")));

    def("contrast_normalization_grad",(void (*)(T&, T&, const T&, const T&, const T&, int, float, float, float, float)) contrast_normalization_grad<V,M,L>, (
                arg("input_gradients"),
                arg("original_outputs"),
                arg("mean_diffs"),
                arg("delta"),
                arg("denoms"),
                arg("patch_size"),
                arg("add_scale"),
                arg("pow_scale"),
                arg("fact_new")=1.f,
                arg("fact_old")=0.f
                ));

    def("response_norm_cross_map",(void (*)(T&, T&, const T&, int, float, float, bool)) response_norm_cross_map<V,M,L>, (
                arg("target"),
                arg("denoms"),
                arg("images"),
                arg("sizeF"),
                arg("add_scale"),
                arg("pow_scale"),
                arg("blocked")));
    def("response_norm_cross_map_grad",(void (*)(T&, T&, const T&, const T&, const T&, int, float, float, bool, float, float)) response_norm_cross_map_grad<V,M,L>, (
                arg("target"),
                arg("denoms"),
                arg("images"),
                arg("sizeF"),
                arg("add_scale"),
                arg("pow_scale"),
                arg("blocked"),
                arg("fact_new")=1.f,
                arg("fact_old")=0.f
                ));
    def("gaussian_blur",(void (*)(T&, const T&, const T&, bool, float, float)) gaussian_blur<V,M,L>, (
                arg("target"),
                arg("images"),
                arg("filter"),
                arg("horiz"),
                arg("fact_new")=1.f,
                arg("fact_old")=0.f));

    def("bed_of_nails",(void (*)(T&, const T&, int, int, float, float)) bed_of_nails<V,M,L>, (
                arg("target"),
                arg("images"),
                arg("start_x"),
                arg("stride_x"),
                arg("fact_new")=1.f,
                arg("fact_old")=0.f));

    def("bed_of_nails_grad",(void (*)(T&, const T&, int, int, float, float)) bed_of_nails_grad<V,M,L>, (
                arg("target"),
                arg("delta"),
                arg("start_x"),
                arg("stride_x"),
                arg("fact_new")=1.f,
                arg("fact_old")=0.f));

    def("crop",(void (*)(T&, const T&, int, int)) crop<V,M,L>, (
                arg("target"),
                arg("images"),
                arg("starty"),
                arg("startx")));
    def("project_to_ball", (void(*)(T&, float)) project_to_ball<V,M,L>, (
                arg("filters"),
                arg("ball_size")));

    def("resize_bilinear", (void(*)(T&, const T&, float)) resize_bilinear<V,M,L>, (
                arg("target"),
                arg("images"),
                arg("scale")));


    
    //def("convolve",(void (*)(M&,M&, M&, int))convolve<typename M::value_type,typename M::memory_space_type, typename M::memory_layout_type>, (
    //            arg("dst"),
    //            arg("img"),
    //            arg("filter"),
    //            arg("nGroups"))
    //   );
    //def("convolve2",(void (*)(M&,M&, M&, int, int))convolve2<typename M::value_type,typename M::memory_space_type, typename M::memory_layout_type>, (
    //            arg("dst"),
    //            arg("img"),
    //            arg("filter"),
    //            arg("numFilters"),
    //            arg("nGroups"))
    //   );
    //def("convolve3",(void (*)(M&,M&, M&, int))convolve3<typename M::value_type,typename M::memory_space_type, typename M::memory_layout_type>, (
    //            arg("dst"),
    //            arg("img"),
    //            arg("filter"),
    //            arg("nGroups"))
    //   );
}

template <class M>
void export_sampling_stuff(){
    typedef typename switch_value_type<M,int>::type Mint;
    //def("super_to_max",(void (*)(M&,M&, int, int, Mint*,M*))super_to_max<typename M::value_type, typename M::memory_space_type, typename M::memory_layout_type, typename M::index_type>, (
                //arg("dst"),
                //arg("img"),
                //arg("poolsize"),
                //arg("overlap"),
                //arg("indices")=object(),
                //arg("filter")=object())
       //);
    //def("subsample",(void (*)(M&,M&, int, bool))subsample<typename M::value_type, typename M::memory_space_type, typename M::memory_layout_type>, (
    //            arg("dst"),
    //            arg("img"),
    //            arg("factor"),
    //            arg("avoidBankConflicts"))
    //   );
    //def("reorder_cpy",(void (*)(M&, M&, int))reorder<typename M::value_type, typename M::memory_space_type, typename M::memory_layout_type>, (
    //            arg("dst_matrix"),
    //            arg("src_matrix"),
    //            arg("block_length")));
    //def("reorder",(void (*)(M&, M&, int))reorder<typename M::value_type, typename M::memory_space_type, typename M::memory_layout_type>, (
    //            arg("dst_matrix"),
    //            arg("src_matrix"),
    //            arg("block_length")));



}

template <class M>
void export_padding_ops(){
    //def("copy_into",(void (*)(M&,M&, int))copy_into<typename M::value_type, typename M::memory_space_type, typename M::memory_layout_type>, (
    //            arg("dst"),
    //            arg("img"),
    //            arg("padding"))
    //   );

    //def("strip_padding",(void (*)(M&,M&, unsigned int))strip_padding<typename M::value_type, typename M::memory_space_type, typename M::memory_layout_type>, (
    //            arg("dst"),
    //            arg("img"),
    //            arg("padding"))
    //   );
}

template <class M, class N, class V>
void export_rlcnp_stuff(){
    //def("row_ncopy",(void (*)(M&,V&, unsigned int))row_ncopy<typename M::value_type, typename M::memory_space_type, typename M::memory_layout_type>, (
    //            arg("dst"),
    //            arg("img"),
    //            arg("rows")));
    //def("cols_ncopy",(void (*)(M&,M&, unsigned int))cols_ncopy<typename M::value_type, typename M::memory_space_type, typename M::memory_layout_type>, (
    //            arg("dst"),
    //            arg("img"),
    //            arg("factor")));
    //def("filter_rotate",(void (*)(M&,M&, unsigned int))filter_rotate<typename M::value_type, typename M::memory_space_type, typename M::memory_layout_type>, (
    //            arg("dst"),
    //            arg("filter"),
    //            arg("fs")));

    //      def("add_maps_h",(void (*)(M&,M&, unsigned int))add_maps_h<typename M::value_type, typename M::memory_space_type, typename M::memory_layout_type,typename M::index_type>, (
    //                                                                                                                      arg("dst"),
    //                                                                                                                      arg("map_matrix"),
    //                                                                                                                      arg("map_size")));
    //def("calc_error_to_blob",(void (*)(M&,M&, M&, unsigned int, unsigned int, float,float, float, float, unsigned int))calc_error_to_blob<typename M::value_type, typename M::memory_space_type, typename M::memory_layout_type>, (
    //            arg("dst"),
    //            arg("img"),
    //            arg("blob_mat"),
    //            arg("image_w"),
    //            arg("image_h"),
    //            arg("sigma_squared"),
    //            arg("temporal_weight"),
    //            arg("interval_size"),
    //            arg("interval_offset"),
    //            arg("window_size")));
    //def("check_exitatory_inhibitory",(void (*)(M&, unsigned int, unsigned int, unsigned int, unsigned int))check_exitatory_inhibitory<typename M::value_type, typename M::memory_space_type, typename M::memory_layout_type>, (
    //            arg("dst"),
    //            arg("start_filter"),
    //            arg("filter_pixels"),
    //            arg("num_exitatory"),
    //            arg("num_inhibitory")));
    //def("init_exitatory_inhibitory",(void (*)(M&, unsigned int, unsigned int, unsigned int, unsigned int))check_exitatory_inhibitory<typename M::value_type, typename M::memory_space_type, typename M::memory_layout_type>, (
    //            arg("dst"),
    //            arg("start_filter"),
    //            arg("filter_pixels"),
    //            arg("num_exitatory"),
    //            arg("num_inhibitory")));
}

void export_convolution_ops(){
    enum_<cuv::alex_conv::pool_type>("pool_type")
        .value("PT_MAX", cuv::alex_conv::PT_MAX)
        .value("PT_AVG", cuv::alex_conv::PT_AVG)
        ;
    export_convolve< tensor<float,host_memory_space,row_major> >();
    export_convolve< tensor<float,dev_memory_space,row_major> >();
    export_sampling_stuff< tensor<float,host_memory_space, row_major> >();
    export_sampling_stuff< tensor<float,dev_memory_space, row_major>  >();
    export_padding_ops< tensor<float,host_memory_space, row_major> >();
    export_padding_ops< tensor<float,host_memory_space, row_major>  >();
    export_padding_ops< tensor<float,dev_memory_space, row_major>  >();
    export_rlcnp_stuff< tensor<float,dev_memory_space, row_major>, tensor<int,dev_memory_space, row_major>, tensor<float,dev_memory_space>  >();
    export_rlcnp_stuff< tensor<float,host_memory_space, row_major>, tensor<int,host_memory_space, row_major>, tensor<float,host_memory_space>  >();
}


