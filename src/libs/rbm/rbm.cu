#include <iostream>
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



		/******************************
		  local connectivity kernel
		 ******************************/
		__global__ void local_connectivity_kernel(float* mat ,int h,int w, int map_h, int map_w, int num_map_hid, int patchsize, int px, int py) {
			const int i = threadIdx.x + blockIdx.x * blockDim.x;
			const int j = threadIdx.y + blockIdx.y * blockDim.y;
			if ((i>=map_h) || (j>=map_w)) return;

			const int  map_hid  = (j * map_h)/map_w;  // scale h
			const int& map_vis  = i;
			const int v_y     = map_vis % py;
			const int h_y     = map_hid % px;
			const int v_x     = map_vis / py;
			const int h_x     = map_hid / px;
			if(abs(v_y-h_y)   > patchsize
					|| abs(v_x-h_x)   > patchsize)
				for(int hidx = i; hidx<h; hidx  += map_h)
					for(int idx = j; idx<num_map_hid*map_w; idx += map_w) // -1: bias
						mat[idx*h + hidx]=0;
			for(int hidx = i; hidx<h; hidx  += map_h)
				for(int b=num_map_hid*map_w; b<w; b += blockDim.y){
					int col = (b+threadIdx.y);
					if(col<w)
						mat[col*h+hidx]=0;
				}
		}

		template<class V, class I>
		void set_local_connectivity_in_dense_matrix(cuv::dense_matrix<V,column_major,dev_memory_space,I>& m, float factor, int patchsize, int px, int py){
			cuvAssert(m.ptr());
			/*int num_maps_lo = (m.h()) / (px*py);*/
			int map_h = px*py;
			int map_w = ceil(map_h * factor);
			static const int bs = 16;
			dim3 blocks(ceil(map_h/(float)bs),ceil(map_w/(float)bs));
			dim3 threads(bs,bs);
			int num_maps = (m.w()) / map_w;
			local_connectivity_kernel<<<blocks,threads>>>(m.ptr(),m.h(),m.w(),map_h,map_w,num_maps, patchsize,px,py);
			cuvSafeCall(cudaThreadSynchronize());
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
template<class __matrix_type>
void set_local_connectivity_in_dense_matrix(__matrix_type& m, float factor, int patchsize, int px, int py){
	detail::set_local_connectivity_in_dense_matrix(m,factor, patchsize, px, py);
}


#define INST(V,L,M,I) \
  template void set_binary_sequence(cuv::dense_matrix<V,L,M,I>& m, const int&); \
  template void sigm_temperature(cuv::dense_matrix<V,L,M,I>& m, const cuv::vector<V,M,I>&); \

INST(float,column_major,host_memory_space,unsigned int);
INST(float,column_major,dev_memory_space,unsigned int);

template
void set_local_connectivity_in_dense_matrix(cuv::dense_matrix<float,column_major,dev_memory_space>& m, float factor, int patchsize, int px, int py);

}
}
}
