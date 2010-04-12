#include <iostream>
#include <stdexcept>
#include <cuda.h>
#include <cutil_inline.h>
#include <thrust/device_ptr.h>
#include <cuv_general.hpp>
#include "vector.hpp"

namespace cuv{

template <class value_type, class index_type>
void alloc( value_type** ptr, index_type size, dev_memory_space) {
	cuvSafeCall(cudaMalloc(ptr, sizeof(value_type)*size));
}

template <class value_type>
void dealloc( value_type** ptr, dev_memory_space) {
	cuvSafeCall(cudaFree(*ptr));
	*ptr = NULL;
}

template <class value_type, class index_type>
void entry_set(value_type* ptr, index_type idx, value_type val, dev_memory_space) {
	thrust::device_ptr<value_type> dev_ptr(ptr);
	dev_ptr[idx]=val;
}

template <class value_type, class index_type>
value_type entry_get(const value_type* ptr, index_type idx, dev_memory_space) {
	const thrust::device_ptr<const value_type> dev_ptr(ptr);
	return (value_type) *(dev_ptr+idx);
}


template class vector<int, dev_memory_space>;
template class vector<float, dev_memory_space>;
template class vector<unsigned char, dev_memory_space>;
template class vector<signed char, dev_memory_space>;

}; // cuv
