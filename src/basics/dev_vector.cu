#include <iostream>
#include <stdexcept>
#include <cuda.h>
#include <cutil_inline.h>

#include <thrust/device_ptr.h>

#include <cuv_general.hpp>

#include "dev_vector.hpp"

namespace cuv{

template<class V,class I>
V dev_vector<V,I>::operator[](size_t idx)const{
	const thrust::device_ptr<const value_type> ptr(this->ptr());
	return (value_type) *(ptr+idx);
}
template<class V,class I>
void
dev_vector<V,I>::alloc(){
		cuvSafeCall(cudaMalloc( (void**)& this->m_ptr, this->memsize() ));
}

template<class V,class I>
void
dev_vector<V,I>::dealloc(){
	  if(this->m_ptr && ! this->m_is_view){
		  cuvSafeCall(cudaFree(this->m_ptr));
			this->m_ptr = NULL;
		}
}

template<class V,class I>
void
dev_vector<V,I>::set(const index_type& i, const value_type& val) {
	thrust::device_ptr<value_type> ptr(this->ptr());
	ptr[i] = val;
}

template class dev_vector<int>;
template class dev_vector<float>;
template class dev_vector<unsigned char>;
template class dev_vector<signed char>;

}
