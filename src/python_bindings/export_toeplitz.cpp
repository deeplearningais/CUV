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
#include <boost/type_traits/is_same.hpp>


#include <toeplitz_matrix.hpp>
#include <filter_factory.hpp>
#include <convert.hpp>

using namespace boost::python;
using namespace cuv;


template<class T>
void
export_toeplitz_common(const char* name){
	typedef T mat;
	typedef typename mat::value_type value_type;
	typedef typename mat::index_type index_type;
	typedef typename mat::vec_type vec_type;

	class_<mat,boost::shared_ptr<mat> > matobj(name);
	matobj
		.def(init<index_type, index_type, int, int, int>())
		.add_property("h", &mat::h)
		.add_property("w", &mat::w)
		.add_property("vec", make_function((vec_type* (mat::*)())(&mat::vec_ptr), return_internal_reference<>()))
		.add_property("num_dia",&mat::num_dia, "number of diagonals")
		.def("__len__",&mat::n, "number of elements")
		.def("alloc",&mat::alloc, "allocate memory")
		.def("dia",(vec_type* (mat::*)(const int&))(&mat::get_dia), "return a view on one of the diagonals", return_value_policy<manage_new_object>())
		.def("dealloc",&mat::dealloc, "deallocate memory")
		.def("__call__",  (const value_type& (mat::*)(const typename mat::index_type&, const typename mat::index_type&)const)(&mat::operator()), return_value_policy<copy_const_reference>()) // igitt.
		;
}

template<class M>
void
export_filter_factory(const char* name){
	typedef M mat;
	typedef typename mat::value_type value_type;
	typedef typename mat::memory_space memory_space;
	typedef typename mat::index_type index_type;

	class_<filter_factory<value_type, memory_space> > (name, init<int, int, int, int, int>())
		//.def("create_toeplitz_from_filters",(toeplitz_matrix<value_type, memory_space>*  (*)(const dense_matrix<value_type, column_major, memory_space>&))
		//                                     &filter_factory<value_type, memory_space>::create_toeplitz_from_filters, (
		//                                                            arg("filter matrix"))
		//                                                            )
		.def("extract_filter",(dense_matrix<value_type, column_major, memory_space>*  (filter_factory<value_type, memory_space>::*)(const dia_matrix<value_type, host_memory_space>&, unsigned int))
				&filter_factory<value_type, memory_space>::extract_filter, (
					arg("dia matrix"), arg("filter number")),
				return_value_policy<manage_new_object>())
		.def("extract_filter",(dense_matrix<value_type, column_major, memory_space>*  (filter_factory<value_type, memory_space>::*)(const dia_matrix<value_type, dev_memory_space>&, unsigned int))
				&filter_factory<value_type, memory_space>::extract_filter, (
					arg("dia matrix"), arg("filter number")),
				return_value_policy<manage_new_object>())
		//.def("extract_filters",(dense_matrix<value_type, column_major, memory_space>*  (*)(const toeplitz_matrix<value_type, memory_space>&))
		//                                    &filter_factory<value_type, memory_space>::extract_filters, (
		//                                                            arg("toeplitz matrix"))
		//                                                            )

		.def("get_dia",(dia_matrix<value_type, memory_space, index_type>*  (filter_factory<value_type, memory_space>::*)())
											&filter_factory<value_type, memory_space>::get_dia,
													"get filter as diagonal matrix",
											return_value_policy<manage_new_object>()
												)
		//.def("get_toeplitz",(toeplitz_matrix<value_type, memory_space, index_type>*  (*)())
		//                                    &filter_factory<value_type, memory_space>::get_toeplitz,
		//                                            "get filter as toeplitz matrix"
		//                                    )
		;
}



void export_toeplitz(){
	export_toeplitz_common<toeplitz_matrix<float,dev_memory_space> >("dev_toeplitz_mat_float");
	export_toeplitz_common<toeplitz_matrix<float,host_memory_space> >("host_toeplitz_mat_float");
	export_filter_factory<filter_factory<float,host_memory_space> >("filter_factory_float");
}
