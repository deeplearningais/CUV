#include <iostream>
#include <stdexcept>
#include <cuda.h>
#include <cutil_inline.h>
#include <cuv_general.hpp>

#include "dev_vector.hpp"

namespace cuv{

template<class V,I>
dev_vector<V,I>::alloc(){
		cuvSafeCall(cudaMalloc( (void**)& this->m_ptr, this->memsize() ));
}
template<class V,I>
dev_vector<V,I>::dealloc(){
	  if(this->m_ptr && ! this->m_is_view){
		  cuvSafeCall(cudaFree(this->m_ptr));
			this->m_ptr = NULL;
		}
}

}
