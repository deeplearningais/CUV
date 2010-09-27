#include "../../basics/dense_matrix.hpp"
#include "rbm.hpp"

namespace cuv{
namespace libs{
namespace rbm{

	namespace detail{

		/* ****************************
		   column_major
	     * ****************************/
		template<class V, class I>
		void set_binary_sequence(cuv::dense_matrix<V,column_major,host_memory_space,I>& m, const int& start){
			const int len = m.h();
			V* ptr  = m.ptr();
			for(unsigned int i=start; i<m.w()+start; i++){
				for(unsigned int j=0;j<len;j++){
					*ptr++ = (i & (1 << (len-1-j))) ? 1 : 0;
				}
			}
		}
		template<class value_type,class index_type>
		__global__
		void set_binary_sequence_kernel(value_type* dst, index_type h, index_type w, int start){
			// for column-major matrices of size (h x w)
			const index_type x = blockIdx.y * blockDim.y + threadIdx.y;
			const index_type y = blockIdx.x * blockDim.x + threadIdx.x;
			if(y>=h) return;
			if(x>=w) return;
			dst[x*h+y] = ((start+x) & (1 << (h-1-y))) ? 1 : 0;
		}
		template<class V, class I>
		void set_binary_sequence(cuv::dense_matrix<V,column_major,dev_memory_space,I>& m, const int& start){
			dim3 threads(16,16);
			dim3 grid(ceil(m.h()/float(threads.x)), ceil(m.w()/float(threads.y)));
			set_binary_sequence_kernel<<<grid,threads>>>(m.ptr(),m.h(),m.w(),start);
			cuvSafeCall(cudaThreadSynchronize());
		}

		template<class V, class I>
		void sigm_temperature(cuv::dense_matrix<V,column_major,host_memory_space,I>& m, const cuv::vector<V,host_memory_space,I>& temp){
			cuvAssert(m.w() == temp.size())
			V* mptr = m.ptr();
			for(unsigned int col=0;col<m.w();col++){
				const V T = temp[col];
				const V* end  = mptr + m.h();
				while(mptr < end){
					*mptr = 1.0/(1.0+exp(-*mptr / T));
					mptr++; 
				}
			}
		}
		template<class value_type,class index_type>
		__global__
		void sigm_temperature_kernel(value_type* dst, const value_type* src, const value_type* temp, index_type h, index_type w){
			// for column-major matrices of size (h x w)
			const index_type x = blockIdx.y * blockDim.y + threadIdx.y;
			const index_type y = blockIdx.x * blockDim.x + threadIdx.x;
			const value_type T = temp[x];
			if(y>=h) return;
			if(x>=w) return;
			dst[x*h+y] = (value_type) (1.0/(1.0 + expf(-src[x*h+y] / T)));
		}
		template<class V, class I>
		void sigm_temperature(cuv::dense_matrix<V,column_major,dev_memory_space,I>& m, const cuv::vector<V,dev_memory_space,I>& temp){
			dim3 threads(16,16);
			dim3 grid(ceil(m.h()/float(threads.x)), ceil(m.w()/float(threads.y)));
			sigm_temperature_kernel<<<grid,threads>>>(m.ptr(),m.ptr(),temp.ptr(),m.h(),m.w());
			cuvSafeCall(cudaThreadSynchronize());
		}

		/* ****************************
		   row_major
	     * ****************************/
		template<class V, class I>
		void set_binary_sequence(cuv::dense_matrix<V,row_major,host_memory_space,I>& m, const int& start){
			// TODO: make column-major view, then call again
		}
		template<class V, class I>
		void set_binary_sequence(cuv::dense_matrix<V,row_major,dev_memory_space,I>& m, const int& start){
			// TODO: make column-major view, then call again
		}

		template<class V, class I>
		void sigm_temperature(cuv::dense_matrix<V,row_major,host_memory_space,I>& m, const cuv::vector<V,host_memory_space,I>& temp){
			// TODO: make column-major view, then call again
		}
		template<class V, class I>
		void sigm_temperature(cuv::dense_matrix<V,row_major,dev_memory_space,I>& m, const cuv::vector<V,dev_memory_space,I>& temp){
			// TODO: make column-major view, then call again
		}
	}

template<class __matrix_type>
void set_binary_sequence(__matrix_type& m, const int& start){
	detail::set_binary_sequence(m,start);
}
template<class __matrix_type,class __vector_type>
void sigm_temperature(__matrix_type& m, const __vector_type& temp){
	detail::sigm_temperature(m,temp);
}


#define INST(V,L,M,I) \
  template void set_binary_sequence(cuv::dense_matrix<V,L,M,I>& m, const int&); \
  template void sigm_temperature(cuv::dense_matrix<V,L,M,I>& m, const cuv::vector<V,M,I>&); \

INST(float,column_major,host_memory_space,unsigned int);
INST(float,column_major,dev_memory_space,unsigned int);

}
}
}
