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
#include <vector>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/type_traits/is_same.hpp>
#include <pyublas/numpy.hpp>


#include <cuv/basics/tensor.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
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
    typename T::reference_type 
    get_reference(T& tens,const boost::python::list &ind){
        typedef typename T::index_type ind_t;
        int length = boost::python::len(ind);
        if (length==1)
            return tens(extract<ind_t>(ind[0]));
        else if (length==2)
            return tens(extract<ind_t>(ind[0]),extract<ind_t>(ind[1]));
        else if (length==3)
            return tens(extract<ind_t>(ind[0]),extract<ind_t>(ind[1]),extract<ind_t>( ind[2]));
        else if (length==4)
            return tens(extract<ind_t>(ind[0]),extract<ind_t>(ind[1]),extract<ind_t>( ind[2]),extract<ind_t>(ind[3]));
        else
            cuvAssert(false);
            //return typename T::reference_type();
       }
    
    template <class T>
    void set(T&tens, const boost::python::list &ind, const typename T::value_type& val){
        get_reference(tens,ind)=val;
    }
    template <class T>
    T* copy(T&old){
        return new T(old);
    }
    template <class T>
    typename T::value_type get(T&tens, const boost::python::list &ind){
        return get_reference(tens,ind);
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
    void reshape(T& tens, const boost::python::list &shape){
        tens.reshape(extract_python_list<typename T::index_type>(shape));
    }
    template <class T>
     boost::python::list shape(T& tens){
         boost::python::list python_shape;
         int n = tens.shape().size();
         for(int i=0; i<n; i++)
             python_shape.append(tens.shape()[i]);
         return python_shape;
    }

    
    /***************************************************
     * Constructing tensors
     ***************************************************/
    template<class V,class M, class L>
    struct basic_tensor_constructor{

	    /// construct using shape 
	    typedef tensor<V,M,L> T;
	    static T* construct_tensor_shape(boost::python::list python_shape){
		    return new T(extract_python_list<typename T::index_type>(python_shape));
	    }
	    /// construct a vector using a single dimension
	    static T* construct_tensor_int(unsigned int len){
		    return new T(len);
	    }
    };
    template <class V,class M, class L>
    struct tensor_constructor : public basic_tensor_constructor<V,M,L> { };

    template <class V, class L>
    struct tensor_constructor<V,host_memory_space,L> : public basic_tensor_constructor<V,host_memory_space,L> { 
	    typedef tensor<V,host_memory_space,L> T;

	    /// construct from numpy, same memory
	    static T* construct_tensor_numpy_array_view(pyublas::numpy_array<typename T::value_type> o){
		    const unsigned int ndim = o.ndim();
		    std::vector<unsigned int> v(ndim);
		    for(int i=0;i<ndim;i++)
			    v[i]=o.dims()[i];
		    PyArrayObject* po = (PyArrayObject*)o.handle().get();
		    cuvAssert(PyArray_ISFARRAY(po) || PyArray_ISCARRAY(po));
		    bool is_f_contiguous = PyArray_ISFARRAY((PyArrayObject *)o.handle().get());
		    if(IsSame<L,column_major>::Result::value != is_f_contiguous)
			    std::reverse(v.begin(),v.end());
		    return new T(v,o.data());
	    }

	    /// construct from numpy, copy memory
	    template<class arg_val_type>
	    static T* construct_tensor_numpy_array_copy(pyublas::numpy_array<arg_val_type> o){
		    const unsigned int ndim = o.ndim();
		    std::vector<unsigned int> v(ndim);
		    for(int i=0;i<ndim;i++)
			    v[i]=o.dims()[i];
		    PyArrayObject* po = (PyArrayObject*)o.handle().get();
		    cuvAssert(PyArray_ISFARRAY(po) || PyArray_ISCARRAY(po));
		    bool is_f_contiguous = PyArray_ISFARRAY(po);
		    if(IsSame<L,column_major>::Result::value != is_f_contiguous)
			    std::reverse(v.begin(),v.end());
		    T* cpy = new T(v);
		    std::copy(o.data(), o.data()+cpy->size(), cpy->ptr());
		    return cpy;
	    }
    };

    template <class V, class L>
    struct tensor_constructor<V,dev_memory_space,L> : public basic_tensor_constructor<V,dev_memory_space,L> { 
	    typedef tensor<V,dev_memory_space,L> T;

	    /// construct from numpy, same memory (invalid for device!)
	    static T* construct_tensor_numpy_array_view(pyublas::numpy_array<typename T::value_type> o){
		    cuvAssert(false);
	    }

	    /// construct from numpy, copy memory
	    template<class arg_val_type>
	    static T* construct_tensor_numpy_array_copy(pyublas::numpy_array<arg_val_type> o){
		    const unsigned int ndim = o.ndim();
		    std::vector<unsigned int> v(ndim);
		    for(int i=0;i<ndim;i++)
			    v[i]=o.dims()[i];
		    PyArrayObject* po = (PyArrayObject*)o.handle().get();
		    cuvAssert(PyArray_ISFARRAY(po) || PyArray_ISCARRAY(po));
		    bool is_f_contiguous = PyArray_ISFARRAY(po);
		    if(IsSame<L,column_major>::Result::value != is_f_contiguous)
			    std::reverse(v.begin(),v.end());

		    if(IsSame<arg_val_type,V>::Result::value){
			    // argument type and value type are the same
			    // we simply need to create a copy of the encapsulated memory on the device
			    tensor<V,host_memory_space,L> view(v,(V*)o.data()); // cast does nothing, since types are always equal at runtime
			    return new T(view);
		    }else{
			    // the argument type and the value type do not match
			    // we therefore first copy everything to a host tensor of the target type
			    // and then create the device tensor by copying this intermediate tensor.
			    typename switch_memory_space_type<T,host_memory_space>::type cpy(v);
			    std::copy(o.data(), o.data()+cpy.size(), cpy.ptr());
			    return new T(cpy);
		    }
	    }
    };


    /***************************************************
     *  Convert Tensor to Numpy
     ***************************************************/
    template<class V,class M, class L>
    struct basic_tens2npy{
	    typedef tensor<V,M,L> T;
	    static boost::python::handle<> create_numpy_array_with_shape(const T& t){
		    std::vector<npy_intp> dims(t.shape().size());
		    std::copy(t.shape().begin(),t.shape().end(), dims.begin());

		    boost::python::handle<> result;
		    if (IsSame<L,row_major>::Result::value) {
			    std::cout << "row major create_numpy_array_with_shape "<<std::endl;
			    result = boost::python::handle<>(PyArray_New(
						    &PyArray_Type, t.shape().size(), &dims[0], 
						    pyublas::get_typenum(V()), 
						    /*strides*/0, 
						    /*data*/NULL,
						    /* ? */ 0, 
						    NPY_INOUT_ARRAY, NULL));
		    }
		    else {
			    std::cout << "col major create_numpy_array_with_shape "<<std::endl;
			    result = boost::python::handle<>(PyArray_New(
						    &PyArray_Type, t.shape().size(), &dims[0], 
						    pyublas::get_typenum(V()), 
						    /*strides*/0, 
						    /*data*/NULL,
						    /* ? */ 0, 
						    NPY_INOUT_FARRAY, NULL));
		    }
		    std::cout << "FARRAY: "<< (PyArray_FLAGS((PyArrayObject*)result.get())&NPY_F_CONTIGUOUS)<<std::endl;
		    std::cout << "CARRAY: "<<(PyArray_FLAGS((PyArrayObject*)result.get())&NPY_C_CONTIGUOUS)<<std::endl;
		    return result;
	    }
    };

    template<class V,class M, class L>
    struct tens2npy : public basic_tens2npy<V,M,L>{ };

    template<class V, class L>
    struct tens2npy<V,host_memory_space,L> : public basic_tens2npy<V,host_memory_space,L>{
	    typedef tensor<V,host_memory_space,L> T;
	    typedef basic_tens2npy<V,host_memory_space,L> my_type;

	    /// copy host vector into a numpy array
	    static boost::python::handle<> to_numpy_copy(const T& t){
		    std::cout << "to_numpy_copy host"<<std::endl;
		    boost::python::handle<> result = my_type::create_numpy_array_with_shape(t);
		    //memcpy((V*)PyArray_DATA((PyArrayObject*)result.get()),t.ptr(),t.memsize());
		    return result;
	    }
    };

    template<class V, class L>
    struct tens2npy<V,dev_memory_space,L> : public basic_tens2npy<V,dev_memory_space,L>{
	    typedef tensor<V,dev_memory_space,L> T;
	    typedef basic_tens2npy<V,dev_memory_space,L> my_type;
	    
	    /// copy device vector into a numpy array
	    static pyublas::numpy_array<V> to_numpy_copy(const T& o){
		    std::cout << "to_numpy_copy device"<<std::endl;
		    boost::python::handle<> result = my_type::create_numpy_array_with_shape(o);
		    tensor<V,host_memory_space,L> t(o.shape(),(V*)PyArray_DATA((PyArrayObject*)result.get())); // view on numpy matrix
		    t = o;  // pull from device; should simply copy the memory
		    return result;
	    }
    };
    
};

template<class T>
void
export_tensor_common(const char* name){
	typedef T arr;
	typedef typename arr::value_type value_type;
	typedef typename arr::memory_space_type memspace_type;
	typedef typename arr::memory_layout_type memlayout_type;
	boost::python::self_t s = boost::python::self;

	class_<arr> c(name);
	c
		.def("__init__", make_constructor(&python_wrapping::tensor_constructor<value_type,memspace_type,memlayout_type>::construct_tensor_shape))
		.def("__init__", make_constructor(&python_wrapping::tensor_constructor<value_type,memspace_type,memlayout_type>::construct_tensor_int))
		.def("__init__", make_constructor(&python_wrapping::tensor_constructor<value_type,memspace_type,memlayout_type>::template construct_tensor_numpy_array_copy<value_type>))
		.def("__init__", make_constructor(&python_wrapping::tensor_constructor<value_type,memspace_type,memlayout_type>::template construct_tensor_numpy_array_copy<double>))
                .def("__len__",&arr::size, "tensor size")
                .def("dealloc",&arr::dealloc, "deallocate memory")
                .def("set",    &python_wrapping::set<T>, "set index to value")
                .def("get",    &python_wrapping::get<T>, "set index to value")
                .def("copy",    &python_wrapping::copy<T>, "get copy of object",return_value_policy<manage_new_object>())
                .def("reshape",    &python_wrapping::reshape<T>, "reshape tensor in place")
                .add_property("np", &python_wrapping::tens2npy<value_type,memspace_type,memlayout_type>::to_numpy_copy)
                .add_property("size", &arr::size)
                .add_property("shape", &python_wrapping::shape<T>, "get shape of tensor")
                .add_property("memsize",&arr::memsize, "size of tensor in memory (bytes)")
		
		.def(s += value_type())
		.def(s -= value_type())
		.def(s *= value_type())
		.def(s /= value_type())
		.def(s += s)
		.def(s -= s)
		.def(s *= s)
		.def(s /= s)

		.def("__add__", ( arr (*) (const arr&,const arr&))operator+<value_type,memspace_type,memlayout_type>)
		.def("__sub__", ( arr (*) (const arr&,const arr&))operator-<value_type,memspace_type,memlayout_type>)
		//.def(s + s) // incompatible with pyublas. god knows why.
		//.def(s - s) // incompatible with pyublas. god knows why.
		.def(s * s)
		.def(s / s)
		.def("__add__", ( arr (*) (const arr&,const value_type&))operator+<value_type,memspace_type,memlayout_type>)
		.def("__sub__", ( arr (*) (const arr&,const value_type&))operator-<value_type,memspace_type,memlayout_type>)
		//.def(s + value_type()) // incompatible with pyublas. god knows why.
		//.def(s - value_type()) // incompatible with pyublas. god knows why.
		.def(s * value_type())
		.def(s / value_type())
		.def("__neg__", ( arr (*) (const arr&))operator-<value_type,memspace_type,memlayout_type>)
		//.def(-s) // incompatible with pyublas. god knows why.
		;
	if(IsSame<memspace_type,host_memory_space>::Result::value){
		def("numpy_view",&python_wrapping::tensor_constructor<value_type,memspace_type,memlayout_type>::construct_tensor_numpy_array_view,
				return_value_policy<manage_new_object, with_custodian_and_ward_postcall<1, 0> >());
	}

	def("this_ptr", this_ptr<arr>);
	def("internal_ptr", internal_ptr<arr>);
	
}

void export_tensor(){
        // row major
	export_tensor_common<tensor<float,dev_memory_space> >("dev_tensor_float");
	export_tensor_common<tensor<float,host_memory_space> >("host_tensor_float");

	export_tensor_common<tensor<unsigned char,dev_memory_space> >("dev_tensor_uc");
	export_tensor_common<tensor<unsigned char,host_memory_space> >("host_tensor_uc");

	export_tensor_common<tensor<int,dev_memory_space> >("dev_tensor_int");
	export_tensor_common<tensor<int,host_memory_space> >("host_tensor_int");

	export_tensor_common<tensor<unsigned int,dev_memory_space> >("dev_tensor_uint");
	export_tensor_common<tensor<unsigned int,host_memory_space> >("host_tensor_uint");
        //column major
	export_tensor_common<tensor<float,dev_memory_space,column_major> >("dev_tensor_float_cm");
	export_tensor_common<tensor<float,host_memory_space,column_major> >("host_tensor_float_cm");

	export_tensor_common<tensor<unsigned char,dev_memory_space,column_major> >("dev_tensor_uc_cm");
	export_tensor_common<tensor<unsigned char,host_memory_space,column_major> >("host_tensor_uc_cm");

	export_tensor_common<tensor<int,dev_memory_space,column_major> >("dev_tensor_int_cm");
	export_tensor_common<tensor<int,host_memory_space,column_major> >("host_tensor_int_cm");

	export_tensor_common<tensor<unsigned int,dev_memory_space,column_major> >("dev_tensor_uint_cm");
	export_tensor_common<tensor<unsigned int,host_memory_space,column_major> >("host_tensor_uint_cm");
	}

