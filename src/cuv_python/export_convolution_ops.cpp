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


//template<class MS, class V,class M, class I>
//struct ms_type {
//};
//template<class V,class M, class I>
//struct ms_type<dev_memory_space,V,M,I> {
        //typedef dev_dense_matrix<V,M,I> type;
//};
//template<class V,class M, class I>
//struct ms_type<host_memory_space,V,M,I> {
        //typedef host_dense_matrix<V,M,I> type;
//};
//template<class Mat, class NewVT>
//struct switch_value_type{
        //typedef typename ms_type<typename matrix_traits<Mat>::memory_space_type,NewVT, typename Mat::memory_layout, typename Mat::index_type>::type type;
//};
//// end: to be refactored


template <class M>
void export_convolve(){
    def("convolve",(void (*)(M&,M&, M&, int))convolve<typename M::value_type,typename M::memory_space_type, typename M::memory_layout,typename M::index_type>, (
                arg("dst"),
                arg("img"),
                arg("filter"),
                arg("nGroups"))
       );
    def("convolve2",(void (*)(M&,M&, M&, int, int))convolve2<typename M::value_type,typename M::memory_space_type, typename M::memory_layout,typename M::index_type>, (
                arg("dst"),
                arg("img"),
                arg("filter"),
                arg("numFilters"),
                arg("nGroups"))
       );
    def("convolve3",(void (*)(M&,M&, M&, int))convolve3<typename M::value_type,typename M::memory_space_type, typename M::memory_layout,typename M::index_type>, (
                arg("dst"),
                arg("img"),
                arg("filter"),
                arg("nGroups"))
       );
}

template <class M>
void export_sampling_stuff(){
    typedef typename switch_value_type<M,int>::type Mint;
    //def("super_to_max",(void (*)(M&,M&, int, int, Mint*,M*))super_to_max<typename M::value_type, typename M::memory_space_type, typename M::memory_layout, typename M::index_type>, (
                //arg("dst"),
                //arg("img"),
                //arg("poolsize"),
                //arg("overlap"),
                //arg("indices")=object(),
                //arg("filter")=object())
       //);
    def("subsample",(void (*)(M&,M&, int, bool))subsample<typename M::value_type, typename M::memory_space_type, typename M::memory_layout>, (
                arg("dst"),
                arg("img"),
                arg("factor"),
                arg("avoidBankConflicts"))
       );
    def("reorder_cpy",(void (*)(M&, M&, int))reorder<M>, (
                arg("dst_matrix"),
                arg("src_matrix"),
                arg("block_length")));
    def("reorder",(void (*)(M&, M&, int))reorder<M>, (
                arg("dst_matrix"),
                arg("src_matrix"),
                arg("block_length")));



}

template <class M>
void export_padding_ops(){
    def("copy_into",(void (*)(M&,M&, int))copy_into<typename M::value_type, typename M::memory_space_type, typename M::memory_layout, typename M::index_type>, (
                arg("dst"),
                arg("img"),
                arg("padding"))
       );

    def("strip_padding",(void (*)(M&,M&, unsigned int))strip_padding<typename M::value_type, typename M::memory_space_type, typename M::memory_layout, typename M::index_type>, (
                arg("dst"),
                arg("img"),
                arg("padding"))
       );
}

template <class M, class N, class V>
void export_rlcnp_stuff(){
    def("row_ncopy",(void (*)(M&,V&, unsigned int))row_ncopy<typename M::value_type, typename M::memory_space_type, typename M::memory_layout, typename M::index_type>, (
                arg("dst"),
                arg("img"),
                arg("rows")));
    def("cols_ncopy",(void (*)(M&,M&, unsigned int))cols_ncopy<typename M::value_type, typename M::memory_space_type, typename M::memory_layout, typename M::index_type>, (
                arg("dst"),
                arg("img"),
                arg("factor")));
    def("filter_rotate",(void (*)(M&,M&, unsigned int))filter_rotate<typename M::value_type, typename M::memory_space_type, typename M::memory_layout,typename M::index_type>, (
                arg("dst"),
                arg("filter"),
                arg("fs")));

    //      def("add_maps_h",(void (*)(M&,M&, unsigned int))add_maps_h<typename M::value_type, typename M::memory_space_type, typename M::memory_layout,typename M::index_type>, (
    //                                                                                                                      arg("dst"),
    //                                                                                                                      arg("map_matrix"),
    //                                                                                                                      arg("map_size")));
    def("calc_error_to_blob",(void (*)(M&,M&, M&, unsigned int, unsigned int, float,float, float, float, unsigned int))calc_error_to_blob<typename M::value_type, typename M::memory_space_type, typename M::memory_layout,typename M::index_type>, (
                arg("dst"),
                arg("img"),
                arg("blob_mat"),
                arg("image_w"),
                arg("image_h"),
                arg("sigma_squared"),
                arg("temporal_weight"),
                arg("interval_size"),
                arg("interval_offset"),
                arg("window_size")));
    def("check_exitatory_inhibitory",(void (*)(M&, unsigned int, unsigned int, unsigned int, unsigned int))check_exitatory_inhibitory<typename M::value_type, typename M::memory_space_type, typename M::memory_layout,typename M::index_type>, (
                arg("dst"),
                arg("start_filter"),
                arg("filter_pixels"),
                arg("num_exitatory"),
                arg("num_inhibitory")));
    def("init_exitatory_inhibitory",(void (*)(M&, unsigned int, unsigned int, unsigned int, unsigned int))check_exitatory_inhibitory<typename M::value_type, typename M::memory_space_type, typename M::memory_layout,typename M::index_type>, (
                arg("dst"),
                arg("start_filter"),
                arg("filter_pixels"),
                arg("num_exitatory"),
                arg("num_inhibitory")));
}

void export_convolution_ops(){
    export_convolve< dense_matrix<float,host_memory_space,row_major> >();
    export_convolve< dense_matrix<float,dev_memory_space,row_major> >();
    export_sampling_stuff< dense_matrix<float,host_memory_space, row_major> >();
    export_sampling_stuff< dense_matrix<float,dev_memory_space, row_major>  >();
    export_padding_ops< dense_matrix<float,host_memory_space, row_major> >();
    export_padding_ops< dense_matrix<float,host_memory_space, row_major>  >();
    export_padding_ops< dense_matrix<float,dev_memory_space, row_major>  >();
    export_rlcnp_stuff< dense_matrix<float,dev_memory_space, row_major>, dense_matrix<int,dev_memory_space, row_major>, tensor<float,dev_memory_space>  >();
    export_rlcnp_stuff< dense_matrix<float,host_memory_space, row_major>, dense_matrix<int,host_memory_space, row_major>, tensor<float,host_memory_space>  >();
}


