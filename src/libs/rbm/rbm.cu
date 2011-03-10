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

		template<class V, class I>
		void set_local_connectivity_in_dense_matrix(cuv::dense_matrix<V,column_major,dev_memory_space,I>& m, int patchsize, int px, int py, int pxh, int pyh, int maxdist_from_main_dia, bool round){
			cuvAssert(m.ptr());
			cuvAssert(m.h()%(px*py) == 0);
			cuvAssert(m.w()%(pxh*pyh) == 0);
			cuvAssert(px/(float)pxh - py/(float)pyh < 0.00001)
			int pix_v = px*py;
			int pix_h = pxh*pyh;
			static const int bs = 16;
			dim3 blocks(ceil(pix_v/(float)bs),ceil(pix_h/(float)bs));
			dim3 threads(bs,bs);
			int num_maps_v = (m.h()) / pix_v;
			int num_maps_h = (m.w()) / pix_h;
			float ratio  = px/(float)pxh;
/*#define V(X) #X << "="<<(X)<<", "*/
			/*std::cout << V(num_maps_h) << V(num_maps_v)<< V(pix_v)<<V(pix_h)<<V(patchsize)<<V(px)<<V(py)<<V(m.h())<<V(m.w())<<std::endl;*/
			if(round)
				local_connectivity_kernel_round<<<blocks,threads>>>(m.ptr(),m.h(),m.w(),pix_v,pix_h,num_maps_h,num_maps_v,patchsize,px,ratio,maxdist_from_main_dia);
			else
				local_connectivity_kernel_square<<<blocks,threads>>>(m.ptr(),m.h(),m.w(),pix_v,pix_h,num_maps_h,num_maps_v,patchsize,px,ratio,maxdist_from_main_dia);
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
		template<class VM, class I>
		void copy_redblack(cuv::dense_matrix<VM,column_major,dev_memory_space,I>& dst, const cuv::dense_matrix<VM,column_major,dev_memory_space,I>& src, const unsigned int num_maps, const unsigned int color){
			cuvAssert(dst.w() == src.w());
			cuvAssert(dst.h() == src.h());
			/*cuvAssert(rowidx.h() == src.w());*/
			dim3 block(16,16);
			dim3 grid(ceil(dst.h()/float(block.y)),
					  ceil(dst.w()/float(block.x)));

			const unsigned int px = sqrt(dst.h() / num_maps);
			copy_redblack_kernel<<<grid,block>>>(dst.ptr(), src.ptr(), dst.h(), dst.w(), px, (bool)color);
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
		template<class VM, class VV, class I>
		void copy_at_rowidx(cuv::dense_matrix<VM,column_major,dev_memory_space,I>& dst, const cuv::dense_matrix<VM,column_major,dev_memory_space,I>& src, const cuv::dense_matrix<VV,column_major,dev_memory_space,I>& rowidx, const unsigned int color){
			cuvAssert(dst.w() == src.w());
			cuvAssert(dst.h() == src.h());
			/*cuvAssert(rowidx.h() == src.w());*/
			dim3 block(16,16);
			dim3 grid(ceil(dst.h()/float(block.y)),
					  ceil(dst.w()/float(block.x)));

			copy_at_rowidx_kernel<<<grid,block>>>(dst.ptr(), src.ptr(), rowidx.ptr(), dst.h(), dst.w(), (bool)color);
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
void set_local_connectivity_in_dense_matrix(__matrix_type& m, int patchsize, int px, int py, int pxh, int pyh, int maxdist_from_main_dia, bool round){
	detail::set_local_connectivity_in_dense_matrix(m,patchsize, px, py, pxh, pyh, maxdist_from_main_dia, round);
}
template<class __matrix_type,class __matrix_type2>
void copy_at_rowidx(__matrix_type& dst, const __matrix_type&  src, const __matrix_type2& rowidx, const unsigned int offset){
	detail::copy_at_rowidx(dst,src,rowidx, offset);
}
template<class __matrix_type>
void copy_redblack(__matrix_type& dst, const __matrix_type&  src, const unsigned int num_maps, const unsigned int color){
	detail::copy_redblack(dst,src, num_maps, color);
}
template
void copy_at_rowidx(cuv::dense_matrix<float,column_major,dev_memory_space>&, const cuv::dense_matrix<float,column_major,dev_memory_space>&, const cuv::dense_matrix<float,column_major,dev_memory_space>&, const unsigned int);


#define INST(V,L,M,I) \
  template void set_binary_sequence(cuv::dense_matrix<V,L,M,I>& m, const int&); \
  template void sigm_temperature(cuv::dense_matrix<V,L,M,I>& m, const cuv::vector<V,M,I>&); \

INST(float,column_major,host_memory_space,unsigned int);
INST(float,column_major,dev_memory_space,unsigned int);

template
void set_local_connectivity_in_dense_matrix(cuv::dense_matrix<float,column_major,dev_memory_space>& m, int patchsize, int px, int py, int,int,int, bool);
template
void copy_redblack(cuv::dense_matrix<float,column_major,dev_memory_space>&, const cuv::dense_matrix<float,column_major,dev_memory_space>&, const unsigned int num_maps, const unsigned int);

}
}
}

