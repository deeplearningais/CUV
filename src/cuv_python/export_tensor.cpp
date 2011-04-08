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

#include <vector>
#include <cuv/basics/tensor.hpp>
#include <cuv/convert/convert.hpp>

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

namespace python_wrapping {
    template <class T>
    typename T::reference_type get_reference(T& tensor,const boost::python::list &ind){
        typedef typename T::index_type ind_t;
        int length = boost::python::len(ind);
        if (length==1)
            return tensor(extract<ind_t>(ind[0]));
        else if (length==2)
            return tensor(extract<ind_t>(ind[0]),extract<ind_t>(ind[1]));
        else if (length==3)
            return tensor(extract<ind_t>(ind[0]),extract<ind_t>(ind[1]),extract<ind_t>( ind[2]));
        else if (length==4)
            return tensor(extract<ind_t>(ind[0]),extract<ind_t>(ind[1]),extract<ind_t>( ind[2]),extract<ind_t>(ind[3]));
        else
            cuvAssert(false);
            //return typename T::reference_type();
       }
    
    template <class T>
    void set(T&tensor, const boost::python::list &ind, const typename T::value_type& val){
        get_reference(tensor,ind)=val;
    }
    template <class T>
    typename T::value_type get(T&tensor, const boost::python::list &ind){
        return get_reference(tensor,ind);
    }
    
    template <class value_type>
        std::vector<value_type> extract_python_list(const boost::python::list & mylist){
            std::vector<value_type> stl_vector;
            int n = boost::python::len(mylist);
            for (int it=0; it < n; it++)
                stl_vector.push_back(extract<value_type>(mylist[it]));
            return stl_vector;
        }

    template <class T>
    void reshape(T& tensor, const boost::python::list &shape){
        tensor.reshape(extract_python_list<typename T::index_type>(shape));
    }
    template <class T>
     boost::python::list shape(T& tensor){
         boost::python::list python_shape;
         int n = tensor.shape().size();
         for(int i=0; i<n; i++){
             python_shape.append(tensor.shape()[i]);
         }
         std::cout << "AAAAAA"<<std::endl;
         return python_shape;
    }
    
};

template<class T>
void
export_tensor_common(const char* name){
	typedef T arr;
	typedef typename arr::value_type value_type;

	class_<arr> (name, init<int>())
                .def("__len__",&arr::size, "tensor size")
                .def("alloc",&arr::allocate, "allocate memory")
                .def("dealloc",&arr::dealloc, "deallocate memory")
                .def("set",    &python_wrapping::set<T>, "set index to value")
                .def("get",    &python_wrapping::get<T>, "set index to value")
                .def("reshape",    &python_wrapping::reshape<T>, "reshape tensor in place")
		//.def("shape",    &python_wrapping::shape<T>, "get shape of tensor",return_value_policy<manage_new_object>())
		.def("shape",    &python_wrapping::shape<T>, "get shape of tensor")
                //.def("shape",    &arr::shape, "get shape of tensor",return_value_policy<copy_const_reference>())
                .add_property("size", &arr::size)
                .add_property("memsize",&arr::memsize, "size of tensor in memory (bytes)")
		;
	def("this_ptr", this_ptr<arr>);
	def("internal_ptr", internal_ptr<arr>);
	
}

template <class T>
void
export_tensor_conversion(){
	def("convert", (void(*)(tensor<T,dev_memory_space>&,const tensor<T,host_memory_space>&)) cuv::convert);
	def("convert", (void(*)(tensor<T,host_memory_space>&,const tensor<T,dev_memory_space>&)) cuv::convert);
}


void export_tensor(){
	export_tensor_common<tensor<float,dev_memory_space> >("dev_tensor_float");
	export_tensor_common<tensor<float,host_memory_space> >("host_tensor_float");

	export_tensor_common<tensor<unsigned char,dev_memory_space> >("dev_tensor_uc");
	export_tensor_common<tensor<unsigned char,host_memory_space> >("host_tensor_uc");

	export_tensor_common<tensor<int,dev_memory_space> >("dev_tensor_int");
	export_tensor_common<tensor<int,host_memory_space> >("host_tensor_int");

	export_tensor_common<tensor<unsigned int,dev_memory_space> >("dev_tensor_uint");
	export_tensor_common<tensor<unsigned int,host_memory_space> >("host_tensor_uint");

	export_tensor_conversion<float>();
	export_tensor_conversion<unsigned char>();
	export_tensor_conversion<int>();
	export_tensor_conversion<unsigned int>();
	}

