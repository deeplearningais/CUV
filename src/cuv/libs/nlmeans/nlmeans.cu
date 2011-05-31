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

#include <cuv/tools/cuv_general.hpp>
#include <cuv/tools/progressbar.hpp>
#include <cuv/basics/cuda_array.hpp>
#include <cuv/basics/tensor.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include "nlmeans.hpp"

namespace cuv{
namespace libs{
namespace nlmeans{

	texture<float,         2, cudaReadModeElementType> cuda_array_tex_float2d; 
	texture<unsigned char, 2, cudaReadModeElementType> cuda_array_tex_uchar2d; 
	texture<float,         3, cudaReadModeElementType> cuda_array_tex_float3d; 
	texture<unsigned char, 3, cudaReadModeElementType> cuda_array_tex_uchar3d; 

	template<int dim, class T> struct texref{ };
	template<> struct texref<2,float>{
		typedef texture<float, 2, cudaReadModeElementType> type;
		static 	inline __device__ __host__ type& get(){ return cuda_array_tex_float2d; }; 
	};
	template<> struct texref<2,unsigned char>{
		typedef texture<unsigned char, 2, cudaReadModeElementType> type;
		static inline __device__ __host__ type& get(){ return cuda_array_tex_uchar2d; }; 
	};
	template<> struct texref<3,float>{
		typedef texture<float, 3, cudaReadModeElementType> type;
		static 	inline __device__ __host__ type& get(){ return cuda_array_tex_float3d; }; 
	};
	template<> struct texref<3,unsigned char>{
		typedef texture<unsigned char, 3, cudaReadModeElementType> type;
		static inline __device__ __host__ type& get(){ return cuda_array_tex_uchar3d; }; 
	};

	template<int search_radius, int filter_radius, class DstT, class SrcT, class I>
	__global__ void get_weights3D(DstT* dst, const SrcT* src, I x, I y, I z, I w, I h, I d){
		/*__shared__ SrcT cmp[2*filter_radius+1][2*filter_radius+1][2*filter_radius+1];*/
		__shared__ SrcT dff[2*filter_radius+1][2*filter_radius+1][2*filter_radius+1];

		dff[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
		/*cmp[threadIdx.z][threadIdx.y][threadIdx.x] =*/
		/*        tex3D(texref<3,SrcT>::get(),*/
		/*                x+threadIdx.x-filter_radius,*/
		/*                y+threadIdx.y-filter_radius,*/
		/*                z+threadIdx.z-filter_radius);*/
		SrcT* cmptr=&dff[0][0][0];
		unsigned int tid = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
		for(int i=x-search_radius+blockIdx.x;i<=x+search_radius;i+=gridDim.x)
			for(int j=y-search_radius+blockIdx.y;j<=y+search_radius;j+=gridDim.y)
				for(int k=z-search_radius+blockIdx.z;k<=z+search_radius;k+=gridDim.z){
					for(int f=-filter_radius+(int)threadIdx.z;f<=filter_radius;f+=blockDim.z){
						if(threadIdx.x<2*filter_radius+1){
							DstT v =  //cmp[threadIdx.z][threadIdx.y][threadIdx.x]
								tex3D(texref<3,SrcT>::get(),
										x+threadIdx.x-filter_radius,
										y+threadIdx.y-filter_radius,
										z+f);
								- tex3D(texref<3,SrcT>::get(),
										i+threadIdx.x-filter_radius,
										j+threadIdx.y-filter_radius,
										k+f);
							dff[threadIdx.z][threadIdx.y][threadIdx.x] += - v*v;
						}
					}
					for (unsigned int offset = blockDim.x*blockDim.y*blockDim.z / 2; offset > 0; offset >>=1) {
					       __syncthreads();
					       if (tid < offset)
						       cmptr[offset] += cmptr[tid+offset];
					}
					if(tid==0)
						dst[blockIdx.x+blockIdx.y*blockDim.y+blockIdx.z*blockDim.y*blockDim.z] = exp(cmptr[0]/(2.f*2.f))*src[x+y*w+z*w*h];
					__syncthreads();
				}
	}
	int divup(int a, int b)
	{
		if (a % b)  /* does a divide b leaving a remainder? */
			return a / b + 1; /* add in additional block */
		else
			return a / b; /* divides cleanly */
	}
	
	template<class T>
	void filter_nlmean(cuv::tensor<T,dev_memory_space,row_major,memory2d_tag>& dst, const cuv::tensor<T,dev_memory_space,row_major,memory2d_tag>& src){
		if(src.ndim()!=3)
			cuvAssert(false);
		if(!equal_shape(dst,src)){
			dst = cuv::tensor<T,dev_memory_space>(src.shape());
		}
		cuda_array<T,dev_memory_space> ca(src.shape()[1],src.shape()[2],src.shape()[0],1);
		ca.assign(src);

		typedef typename texref<3,T>::type textype;
		textype& tex = texref<3,T>::get();
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
		cuvSafeCall(cudaBindTextureToArray(tex, ca.ptr(), channelDesc));
		tex.normalized = false;
		tex.filterMode = cudaFilterModePoint;
		tex.addressMode[0] = cudaAddressModeClamp;
		tex.addressMode[1] = cudaAddressModeClamp;

		static const int filter_radius = 7;
		static const int search_radius = 5;
		unsigned int w = src.shape()[2], h=src.shape()[1], d=src.shape()[0];
		dim3 blocks(divup(filter_radius,4),divup(filter_radius,4), divup(filter_radius,4));
		dim3 threads(16*divup(2*filter_radius+1,16),2*filter_radius+1,1);
		ProgressBar pb(w*h*d);
		cuv::tensor<float,dev_memory_space>weights(extents[blocks.z][blocks.y][blocks.x]);
		for(unsigned int i=0;i<w;i++){
			for(unsigned int j=0;j<h;j++){
				for(unsigned int k=0;k<d;k++){
					get_weights3D<search_radius, filter_radius><<<blocks,threads>>>(weights.ptr(),src.ptr(),i,j,k,w,h,d);
					//cuvSafeCall(cudaThreadSynchronize());
					dst(k,j,i) = (T) sum(weights);
					pb.inc();
				}
			}
			
		}
		cuvSafeCall(cudaUnbindTexture(tex));
	}

	/*template void filter_nlmean(cuv::tensor<float,dev_memory_space>& dst, const cuv::tensor<float,dev_memory_space>& src);*/
	template void filter_nlmean(cuv::tensor<float,dev_memory_space,row_major,memory2d_tag>& dst, const cuv::tensor<float,dev_memory_space,row_major,memory2d_tag>& src);
}
}
}
