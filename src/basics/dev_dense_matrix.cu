#include <iostream>
#include <stdexcept>
#include <cuda.h>
#include <cutil_inline.h>
#include <cuv_general.hpp>

#include <dev_dense_matrix.hpp>

#define DDM0 template<class __value_type, class __mem_layout, class __index_type> \
	dev_dense_matrix<__value_type,__mem_layout, __index_type>
#define DDM(RETURN_TYPE) template<class __value_type, class __mem_layout, class __index_type> \
	RETURN_TYPE dev_dense_matrix<__value_type,__mem_layout, __index_type>



namespace cuv{

	/*
	 * Life Cycle
	 *
	 */
	DDM0::dev_dense_matrix(const index_type& h, const index_type& w)
	: base_type(h,w) { alloc(); }

	DDM0 ::dev_dense_matrix(const index_type& h, const index_type& w, value_type* p, const bool& is_view)
	: base_type(h,w,p,is_view) {}

	DDM(void)::alloc(){
		cuvSafeCall(cudaMalloc( (void**)& this->m_ptr, this->memsize() ));
	}

	DDM0 ::~dev_dense_matrix(){
	  dealloc();
	}
	DDM(void)::dealloc(){
	  if(this->m_ptr && ! this->m_is_view){
		  cuvSafeCall(cudaFree(this->m_ptr));
			this->m_ptr = NULL;
		}
	}

	template class dev_dense_matrix<float,column_major,int>;
	template class dev_dense_matrix<float,row_major,int>;
	template class dev_dense_matrix<unsigned char,row_major,int>;
}
