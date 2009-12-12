#include <stdexcept>

#include <cuda.h>
#include <dev_dense_matrix.hpp>

#define DDM0 template<class __value_type, class __mem_layout, class __index_type> \
	dev_dense_matrix<__value_type,__mem_layout, __index_type>
#define DDM(X) template<class __value_type, class __mem_layout, class __index_type> \
	X dev_dense_matrix<__value_type,__mem_layout, __index_type>



namespace cuv{

	/*
	 * Life Cycle
	 *
	 */
	DDM0::dev_dense_matrix(const index_type& h, const index_type& w)
	: dense_matrix(h,w) { alloc(); }

	DDM0::dev_dense_matrix(const index_type& h, const index_type& w, value_type* p, const bool& is_view)
	: dense_matrix(h,w,p,is_view) {}

	DDM(void)::alloc(const index_type& h, const index_type& w){
		cudaMalloc( (void**)&m_ptr, memsize() );
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) throw std::runtime_error("Device memory allocation failed");
		checkCudaError("allocation");
	}

	DDM0::~dev_dense_matrix(){
	  dealloc();
	}
	DDM(void)::dealloc(){
	  if(m_ptr && !m_is_view){
		  cutilSafeCall(cudaFree(m_ptr));
			m_ptr = NULL;
		}
	}

}
