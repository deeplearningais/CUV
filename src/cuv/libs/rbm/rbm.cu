#include <iostream>
#include <cuv/basics/tensor.hpp>
#include <cuv/libs/rbm/rbm.hpp>

namespace cuv{
namespace libs{
namespace rbm{

	namespace detail{

		/* ****************************
		   column_major
	     * ****************************/
		template<class V>
		void set_binary_sequence(cuv::tensor<V,host_memory_space,column_major>& m, const int& start){
			const int len = m.shape()[0];
			V* ptr  = m.ptr();
			for(unsigned int i=start; i<m.shape()[1]+start; i++){
				for(unsigned int j=0;j<len;j++){
					*ptr++ = (i & (1 << (len-1-j))) ? 1 : 0;
				}
			}
		}
		template<class value_type, class index_type>
		__global__
		void set_binary_sequence_kernel(value_type* dst, index_type h, index_type w, int start){
			// for column-major matrices of size (h x w)
			const index_type x = blockIdx.y * blockDim.y + threadIdx.y;
			const index_type y = blockIdx.x * blockDim.x + threadIdx.x;
			if(y>=h) return;
			if(x>=w) return;
			dst[x*h+y] = ((start+x) & (1 << (h-1-y))) ? 1 : 0;
		}
		template<class V>
		void set_binary_sequence(cuv::tensor<V,dev_memory_space,column_major>& m, const int& start){
			dim3 threads(16,16);
			dim3 grid(ceil(m.shape()[0]/float(threads.x)), ceil(m.shape()[1]/float(threads.y)));
			set_binary_sequence_kernel<<<grid,threads>>>(m.ptr(),m.shape()[0],m.shape()[1],start);
			cuvSafeCall(cudaThreadSynchronize());
		}

		template<class V>
		void sigm_temperature(cuv::tensor<V,host_memory_space,column_major>& m, const cuv::tensor<V,host_memory_space>& temp){
			cuvAssert(m.shape()[1] == temp.size())
			V* mptr = m.ptr();
			for(unsigned int col=0;col<m.shape()[1];col++){
				const V T = temp[col];
				const V* end  = mptr + m.shape()[0];
				while(mptr < end){
					*mptr = 1.0/(1.0+exp(-*mptr / T));
					mptr++; 
				}
			}
		}
		template<class value_type, class index_type>
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
		template<class V>
		void sigm_temperature(cuv::tensor<V,dev_memory_space,column_major>& m, const cuv::tensor<V,dev_memory_space>& temp){
			dim3 threads(16,16);
			dim3 grid(ceil(m.shape()[0]/float(threads.x)), ceil(m.shape()[1]/float(threads.y)));
			sigm_temperature_kernel<<<grid,threads>>>(m.ptr(),m.ptr(),temp.ptr(),m.shape()[0],m.shape()[1]);
			cuvSafeCall(cudaThreadSynchronize());
		}

		/* ****************************
		   row_major
	     * ****************************/
		template<class V>
		void set_binary_sequence(cuv::tensor<V,host_memory_space,row_major>& m, const int& start){
			// TODO: make column-major view, then call again
		}
		template<class V>
		void set_binary_sequence(cuv::tensor<V,dev_memory_space,row_major>& m, const int& start){
			// TODO: make column-major view, then call again
		}

		template<class V>
		void sigm_temperature(cuv::tensor<V,host_memory_space,row_major>& m, const cuv::tensor<V,host_memory_space>& temp){
			// TODO: make column-major view, then call again
		}
		template<class V>
		void sigm_temperature(cuv::tensor<V,dev_memory_space,row_major>& m, const cuv::tensor<V,dev_memory_space>& temp){
			// TODO: make column-major view, then call again
		}



		/******************************
		  local connectivity kernel
		 ******************************/
		template<class T>
		__global__ void local_connectivity_kernel_round(T* mat ,int h,int w, int pix_v, int pix_h, int num_map_hid, int num_map_vis, int patchsize, int px, float ratio, int maxdist_from_main_dia) {
			const int i = threadIdx.x + blockIdx.x * blockDim.x; // i changes with the visible unit
			const int j = threadIdx.y + blockIdx.y * blockDim.y; // j changes with the hidden unit
			if ((i>=pix_v) || (j>=pix_h)) return;

			const int py_hidden = px/ratio;
			const int  v_y      = i % px;
			const int  v_x      = i / px;
			const int  h_y      = ((j % py_hidden) + 0.5)*ratio;
			const int  h_x      = ((j / py_hidden) + 0.5)*ratio;

			const int dist = (v_y-h_y)*(v_y-h_y)+(v_x-h_x)*(v_x-h_x);
			const bool outpatch = dist > patchsize*patchsize; // we are not in the patch
			for(int vidx = 0; vidx<num_map_vis; vidx++) // loop over visible maps
				for(int hidx = 0; hidx<num_map_hid; hidx++)// loop over hidden maps
					if(outpatch || abs(vidx-hidx)>maxdist_from_main_dia)
						mat[(hidx*pix_h+j)*h + vidx*pix_v+i]=(T)0; // reset this value
		}
		template<class T>
		__global__ void local_connectivity_kernel_square(T* mat ,int h,int w, int pix_v, int pix_h, int num_map_hid, int num_map_vis, int patchsize, int px, float ratio, int maxdist_from_main_dia) {
			const int i = threadIdx.x + blockIdx.x * blockDim.x; // i changes with the visible unit
			const int j = threadIdx.y + blockIdx.y * blockDim.y; // j changes with the hidden unit
			if ((i>=pix_v) || (j>=pix_h)) return;

			const int py_hidden = px/ratio;
			const int  v_y      = i % px;
			const int  v_x      = i / px;
			const int  h_y      = ((j % py_hidden) + 0.5)*ratio;
			const int  h_x      = ((j / py_hidden) + 0.5)*ratio;

			const bool outpatch = (    abs(v_y-h_y)   > patchsize
					||                 abs(v_x-h_x)   > patchsize); // we are not in the patch
			for(int vidx = 0; vidx<num_map_vis; vidx++) // loop over visible maps
				for(int hidx = 0; hidx<num_map_hid; hidx++)// loop over hidden maps
					if(outpatch || abs(vidx-hidx)>maxdist_from_main_dia)
						mat[(hidx*pix_h+j)*h + vidx*pix_v+i]=(T)0; // reset this value
		}

		template<class V>
		void set_local_connectivity_in_dense_matrix(cuv::tensor<V,dev_memory_space,column_major>& m, int patchsize, int px, int py, int pxh, int pyh, int maxdist_from_main_dia, bool round){
			cuvAssert(m.ptr());
			cuvAssert(m.shape()[0]%(px*py) == 0);
			cuvAssert(m.shape()[1]%(pxh*pyh) == 0);
			cuvAssert(px/(float)pxh - py/(float)pyh < 0.00001)
			int pix_v = px*py;
			int pix_h = pxh*pyh;
			static const int bs = 16;
			dim3 blocks(ceil(pix_v/(float)bs),ceil(pix_h/(float)bs));
			dim3 threads(bs,bs);
			int num_maps_v = (m.shape()[0]) / pix_v;
			int num_maps_h = (m.shape()[1]) / pix_h;
			float ratio  = px/(float)pxh;
/*#define V(X) #X << "="<<(X)<<", "*/
			/*std::cout << V(num_maps_h) << V(num_maps_v)<< V(pix_v)<<V(pix_h)<<V(patchsize)<<V(px)<<V(py)<<V(m.shape()[0])<<V(m.shape()[1])<<std::endl;*/
			if(round)
				local_connectivity_kernel_round<<<blocks,threads>>>(m.ptr(),m.shape()[0],m.shape()[1],pix_v,pix_h,num_maps_h,num_maps_v,patchsize,px,ratio,maxdist_from_main_dia);
			else
				local_connectivity_kernel_square<<<blocks,threads>>>(m.ptr(),m.shape()[0],m.shape()[1],pix_v,pix_h,num_maps_h,num_maps_v,patchsize,px,ratio,maxdist_from_main_dia);
			cuvSafeCall(cudaThreadSynchronize());
		}


		/**********************************
		  copy redblack
		 **********************************/
		template <class VM, class I>
		__global__
		void copy_redblack_kernel(VM*dst, const VM*src, const I h, const I w, const I px, const bool color){
			unsigned int tidx = threadIdx.y + blockIdx.y*blockDim.y;
			unsigned int tidy = threadIdx.x + blockIdx.x*blockDim.x;
			if(tidy >= h) return;
			if(tidx >= w) return;
			const unsigned int imgx = tidy % px;
			const unsigned int imgy = tidy / px;

			bool need_update = color ^ ((imgx+imgy)%2);
			if(need_update)
				dst[tidx*h+tidy] = src[tidx*h+tidy];
		}
		template<class VM>
		void copy_redblack(cuv::tensor<VM,dev_memory_space,column_major>& dst, const cuv::tensor<VM,dev_memory_space,column_major>& src, const unsigned int num_maps, const unsigned int color){
			cuvAssert(dst.shape()[1] == src.shape()[1]);
			cuvAssert(dst.shape()[0] == src.shape()[0]);
			/*cuvAssert(rowidx.shape()[0] == src.shape()[1]);*/
			dim3 block(16,16);
			dim3 grid(ceil(dst.shape()[0]/float(block.y)),
					  ceil(dst.shape()[1]/float(block.x)));

			const unsigned int px = sqrt(dst.shape()[0] / num_maps);
			copy_redblack_kernel<<<grid,block>>>(dst.ptr(), src.ptr(), dst.shape()[0], dst.shape()[1], px, (bool)color);
			cuvSafeCall(cudaThreadSynchronize());
		}
		/**********************************
		  copy at rowidx
		 **********************************/
		template <class VM, class VV, class I>
		__global__
		void copy_at_rowidx_kernel(VM*dst, const VM*src, const VV* ridx, const I h, const I w, const bool color){
			/*unsigned int tidx = threadIdx.x + blockIdx.x*blockDim.x;*/
			/*if(tidx >= w) return;*/

			/*const unsigned int col_offset = tidx*h;*/
			/*for(unsigned int i=0; i<b; i++){*/
				/*I r   = (I) ridx[w*i+tidx];*/
				/*const unsigned int map_offset = offset*i;*/
				/*dst[col_offset+map_offset + r] = src[col_offset+map_offset + r];*/
			/*}*/

			unsigned int tidx = threadIdx.y + blockIdx.y*blockDim.y;
			unsigned int tidy = threadIdx.x + blockIdx.x*blockDim.x;
			if(tidx >= w) return;
			if(tidy >= h) return;

			bool need_update = color ^ ((tidx+tidy)%2);
			if(need_update)
				dst[tidx*h+tidy] = src[tidx*h+tidy];
		}
		template<class VM, class VV>
		void copy_at_rowidx(cuv::tensor<VM,dev_memory_space,column_major>& dst, const cuv::tensor<VM,dev_memory_space,column_major>& src, const cuv::tensor<VV,dev_memory_space,column_major>& rowidx, const unsigned int color){
			cuvAssert(dst.shape()[1] == src.shape()[1]);
			cuvAssert(dst.shape()[0] == src.shape()[0]);
			/*cuvAssert(rowidx.shape()[0] == src.shape()[1]);*/
			dim3 block(16,16);
			dim3 grid(ceil(dst.shape()[0]/float(block.y)),
					  ceil(dst.shape()[1]/float(block.x)));

			copy_at_rowidx_kernel<<<grid,block>>>(dst.ptr(), src.ptr(), rowidx.ptr(), dst.shape()[0], dst.shape()[1], (bool)color);
			cuvSafeCall(cudaThreadSynchronize());
		}
        __global__ void bitflip_kernel(float* M, int height, int row, int n) {
                const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int off = blockDim.x * gridDim.x;
            for (unsigned int i = idx; i < n; i += off){
                        M[i * height + row] = 1 - M[i * height + row];
                }

        }
	template<class V, class I>
	void bitflip(tensor<V,dev_memory_space,column_major>& m, const I& row){
		int num_threads = (int) min(512.f, ceil((float)sqrt(m.shape()[1])));
		int num_blocks  = (int) ceil((float)m.shape()[1]/num_threads);
		bitflip_kernel<<<num_blocks,num_threads>>>(m.ptr(),m.shape()[0],row, m.shape()[1]);
		cuvSafeCall(cudaThreadSynchronize());
	}
	template<class V, class I>
	void bitflip(tensor<V,host_memory_space,column_major>& m, const I& row){
		for(int i=0;i<m.shape()[1];i++)
			m(row,i)=(V)(1.f-m(row,i));
	}
}
// bitflip a row of a column-major matrix
template<class __value_type, class __memory_layout, class __memory_space_type>
void bitflip(tensor<__value_type,__memory_layout,__memory_space_type>& matrix,
		typename tensor<__value_type,__memory_layout,__memory_space_type>::index_type row){
                cuvAssert(matrix.ndim()==2);
		cuvAssert(row<matrix.shape()[0]);
		cuvAssert(matrix.ptr());
		detail::bitflip(matrix,row);
}

template <class __value_type, class __memory_space_type, class __memory_layout_type>
void set_binary_sequence(tensor<__value_type,__memory_space_type,__memory_layout_type>& m, const int& start){
	detail::set_binary_sequence(m,start);
}

template <class __value_type, class __memory_space_type, class __memory_layout_type>
void sigm_temperature(tensor<__value_type,__memory_space_type,__memory_layout_type>& m, const tensor<__value_type,__memory_space_type>& temp){
	detail::sigm_temperature(m,temp);
}
template <class __value_type, class __memory_space_type, class __memory_layout_type>
void set_local_connectivity_in_dense_matrix(tensor<__value_type,__memory_space_type,__memory_layout_type>& m, int patchsize, int vx, int vy, int hx, int hy, int maxdist_from_main_dia, bool round){
	detail::set_local_connectivity_in_dense_matrix(m,patchsize, vx, vy, hx, hy, maxdist_from_main_dia, round);
}
template <class __value_type, class __memory_space_type, class __memory_layout_type>
void copy_at_rowidx(tensor<__value_type,__memory_space_type,__memory_layout_type>& dst, const tensor<__value_type,__memory_space_type,__memory_layout_type>&  src, const tensor<typename tensor<__value_type,__memory_space_type,__memory_layout_type>::index_type,__memory_space_type, __memory_layout_type>& rowidx, const unsigned int offset){
	detail::copy_at_rowidx(dst,src,rowidx, offset);
}
template <class __value_type, class __memory_space_type, class __memory_layout_type>
void copy_redblack(tensor<__value_type,__memory_space_type,__memory_layout_type>& dst, const tensor<__value_type,__memory_space_type,__memory_layout_type>&  src, const unsigned int num_maps, const unsigned int color){
	detail::copy_redblack(dst,src, num_maps, color);
}

#define INST(V,L,M,I) \
  template void set_binary_sequence(cuv::tensor<V,L,M>& m, const int&); \
  template void sigm_temperature(cuv::tensor<V,L,M>& m, const cuv::tensor<V,L>&); \

INST(float,host_memory_space,column_major,unsigned int);
INST(float,dev_memory_space,column_major,unsigned int);

template
void set_local_connectivity_in_dense_matrix(cuv::tensor<float,dev_memory_space,column_major>& m, int patchsize, int px, int py, int,int,int, bool);
template
void copy_redblack(cuv::tensor<float,dev_memory_space,column_major>&, const cuv::tensor<float,dev_memory_space,column_major>&, const unsigned int num_maps, const unsigned int);
template
void copy_at_rowidx(cuv::tensor<float,dev_memory_space,column_major>&, const cuv::tensor<float,dev_memory_space,column_major>&, const cuv::tensor<unsigned int,dev_memory_space,column_major>&, const unsigned int);
template void bitflip(tensor<float,host_memory_space,column_major>&, unsigned int);
template void bitflip(tensor<float,dev_memory_space,column_major>&, unsigned int);
}
}
}

