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
