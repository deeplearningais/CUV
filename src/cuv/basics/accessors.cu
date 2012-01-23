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
#include <thrust/device_ptr.h>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/linear_memory.hpp>
#include <cuv/tools/meta_programming.hpp>

#include "accessors.hpp"

namespace cuv{

template <class value_type, class size_type>
struct allocator<value_type,size_type,dev_memory_space>{
	void alloc2d( value_type** ptr, size_type& pitch, size_type height, size_type width ) const{
		size_t p;
		cuvSafeCall(cudaMallocPitch(ptr, &p, sizeof(value_type)*width, height));
		pitch = p;
	}
	void alloc( value_type** ptr, size_type size) const{
		cuvSafeCall(cudaMalloc(ptr, sizeof(value_type)*size));
	}
	void dealloc( value_type** ptr) const {
		cuvSafeCall(cudaFree((void*)*ptr));
		*ptr = NULL;
	}
	void alloc(const value_type** ptr, size_type size) const{
	       cuvAssert(false);
	}
	void dealloc(const value_type** ptr)const {
	       cuvAssert(false);
	}
	void copy(value_type* dst, const value_type*src,size_type size, host_memory_space){
		cuvSafeCall(cudaMemcpy( dst, src, size*sizeof( value_type ), cudaMemcpyHostToDevice ));
	}
	template<class value_type2>
	void
	copy(value_type* dst, const value_type2*src,size_type size, dev_memory_space){
		if(IsSame<value_type,value_type2>::Result::value){
			cuvSafeCall(cudaMemcpy( dst, src, size*sizeof( value_type ), cudaMemcpyDeviceToDevice ));
		}
		else{
			thrust::copy(thrust::device_ptr<value_type2>(const_cast<value_type2*>(src)),
					thrust::device_ptr<value_type2>(const_cast<value_type2*>(src)) + size,
					thrust::device_ptr<value_type>(dst));
			cuvSafeCall(cudaThreadSynchronize());
		}
	}
	void copy2d(value_type* dst, const value_type*src,size_type dpitch, size_type spitch, size_type h, size_type w, host_memory_space){
		cuvSafeCall(cudaMemcpy2D(dst,dpitch,src,spitch,w*sizeof(value_type),h,cudaMemcpyHostToDevice));
	}
	void copy2d(value_type* dst, const value_type*src,size_type dpitch, size_type spitch, size_type h, size_type w, dev_memory_space){
		cuvSafeCall(cudaMemcpy2D(dst,dpitch,src,spitch,w*sizeof(value_type),h,cudaMemcpyDeviceToDevice));
	}
};

template<class V,class I>
void
allocator<V,I,host_memory_space>::alloc(V** ptr, I size)const{
	cuvSafeCall(cudaHostAlloc(ptr,size*sizeof(V),cudaHostAllocDefault));
}
template<class V,class I>
void
allocator<V,I,host_memory_space>::dealloc(V** ptr)const{
	cuvSafeCall(cudaFreeHost(*ptr));
	*ptr = 0;
}
template<class V,class I>
void
allocator<V,I,host_memory_space>::alloc2d(V** ptr, I& pitch, I height, I width)const{
	pitch = width*sizeof(V);
	*ptr  = new V[height*width];
}
template<class V,class I>
void
allocator<V,I,host_memory_space>::copy(V*dst, const V*src,I size,dev_memory_space){
	cuvSafeCall(cudaMemcpy( dst, src, size*sizeof( V ), cudaMemcpyDeviceToHost ));
}

template<class V,class I>
void
allocator<V,I,host_memory_space>::copy2d(V* dst, const V*src,I dpitch, I spitch, I h, I w, dev_memory_space){
	cuvSafeCall(cudaMemcpy2D(dst,dpitch,src,spitch,w*sizeof(V),h,cudaMemcpyDeviceToHost));
}

template<class V,class I>
void
allocator<V,I,host_memory_space>::copy2d(V* dst, const V*src,I dpitch, I spitch, I h, I w, host_memory_space){
	cuvSafeCall(cudaMemcpy2D(dst,dpitch,src,spitch,w*sizeof(V),h,cudaMemcpyHostToHost));
}

template <class value_type, class size_type>
void entry_set(value_type* ptr, size_type idx, value_type val, dev_memory_space) {
	thrust::device_ptr<value_type> dev_ptr(ptr);
	dev_ptr[idx]=val;
}

template <class value_type, class size_type>
value_type entry_get(const value_type* ptr, size_type idx, dev_memory_space) {
	const thrust::device_ptr<const value_type> dev_ptr(ptr);
	return (value_type) *(dev_ptr+idx);
}

#define VECTOR_INST(T,I) \
template struct allocator<T, I, dev_memory_space>; \
template struct allocator<T, I, host_memory_space>; \
template void entry_set(T*, I, T, dev_memory_space); \
template T entry_get(const T*, I, dev_memory_space); \
template void allocator<T, I, dev_memory_space>::copy<        float>(T* dst, const         float*src,I size, dev_memory_space); \
template void allocator<T, I, dev_memory_space>::copy<unsigned char>(T* dst, const unsigned char*src,I size, dev_memory_space); \
template void allocator<T, I, dev_memory_space>::copy<  signed char>(T* dst, const signed   char*src,I size, dev_memory_space); \
template void allocator<T, I, dev_memory_space>::copy<          int>(T* dst, const int          *src,I size, dev_memory_space); \
template void allocator<T, I, dev_memory_space>::copy<unsigned  int>(T* dst, const unsigned int *src,I size, dev_memory_space);

VECTOR_INST(float, int);
VECTOR_INST(unsigned char, int);
VECTOR_INST(signed char, int);
VECTOR_INST(int, int);
VECTOR_INST(unsigned int, int);

VECTOR_INST(float, unsigned int);
VECTOR_INST(unsigned char, unsigned int);
VECTOR_INST(signed char, unsigned int);
VECTOR_INST(int, unsigned int);
VECTOR_INST(unsigned int, unsigned int);


}; // cuv
