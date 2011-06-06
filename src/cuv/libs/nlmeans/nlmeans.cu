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

#include <cstdio>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/tools/progressbar.hpp>
#include <cuv/basics/cuda_array.hpp>
#include <cuv/basics/tensor.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>

#include "nlmeans.hpp"
#include "conv3d.hpp"

namespace cuv{
namespace libs{
namespace nlmeans{
#define PITCH(PTR,PITCH,Y,X) ((typeof(PTR))((char*)(PTR) + (PITCH)*(Y)) + (X))

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

	template<class V, class I1>
	__device__ I1 clamp(const V& i, const I1&maxi){
		return ((i<0)?0:((i>=maxi)?(maxi-1):i));
	}
	template<bool weights2d, class DstT, class SrcT, class I, class DI>
	__global__ 
	void mult_offset(DstT* dst, const SrcT* weights, const SrcT* orig, DI x, DI y, DI z, I w, I h, I d, I spitch){
		const int xstart = threadIdx.x +blockIdx.x*blockDim.x;
		const int ystart = threadIdx.y +blockIdx.y*blockDim.y;
		const int zstart = threadIdx.z +blockIdx.z*blockDim.z;
		const int xoff   = blockDim.x*gridDim.x;
		const int yoff   = blockDim.y*gridDim.y;
		const int zoff   = blockDim.z*gridDim.z;
		for(I i=zstart;i<d;i+=zoff)
		for(I j=ystart;j<h;j+=yoff)
		for(I k=xstart;k<w;k+=xoff){
			SrcT wgt = weights2d ? weights[k+j*w] : weights[k+j*w+i*w*h];
			dst[k+j*w+i*w*h] += wgt *
				/**PITCH(orig,spitch,clamp(j+y,h)+clamp(i+z,d)*h, clamp(k+x,w) );*/
				/*orig[clamp(j+y,h)*w + clamp(i+z,d)*w*h + clamp(k+x,w)];*/
				tex3D(texref<3,SrcT>::get(), k+x, j+y, i+z);
		}
	}
	template<class DstT, class SrcT, class I, class DI>
	__global__ 
	void get_sqdiff2d(DstT* diffs, const SrcT* src, DI x, DI y, DI z, I w, I h, I d, I spitch){
		const int xstart = threadIdx.x +blockIdx.x*blockDim.x;
		const int ystart = threadIdx.y +blockIdx.y*blockDim.y;
		const int xoff   = blockDim.x*gridDim.x;
		const int yoff   = blockDim.y*gridDim.y;
		for(int j=ystart;j<h;j+=yoff)
			for(int k=xstart;k<w;k+=xoff){
				DstT res = 0.0f;
				for(int i=0;i<d;i+=1){
					/*
					 *DstT v = src[j*w+i*h*w+k]
					 *        -src[clamp(j+y,h)*w+clamp(i+z,d)*w*h+ clamp(k+x,w)];
					 */
					/*
					 *DstT v = *PITCH(src,spitch,j+i*h,k)
					 *        -*PITCH(src,spitch,clamp(j+y,h)+clamp(i+z,d)*h, clamp(k+x,w) );
					 */
					DstT v =  tex3D(texref<3,SrcT>::get(), k, j, i)
						- tex3D(texref<3,SrcT>::get(), k+x,j+y,i+z);
					;
					v = ((k+x>=w) | (j+y>=h)) ? 1E6 : v;
					res += v*v;
				}
				diffs[k+j*w] = res/d;
			}
	}
	template<class DstT, class SrcT, class I, class DI>
	__global__ 
	void get_sqdiff(DstT* diffs, const SrcT* src, DI x, DI y, DI z, I w, I h, I d, I spitch){
		const int xstart = threadIdx.x +blockIdx.x*blockDim.x;
		const int ystart = threadIdx.y +blockIdx.y*blockDim.y;
		const int zstart = threadIdx.z +blockIdx.z*blockDim.z;
		const int xoff   = blockDim.x*gridDim.x;
		const int yoff   = blockDim.y*gridDim.y;
		const int zoff   = blockDim.z*gridDim.z;
		for(int i=zstart;i<d;i+=zoff)
		for(int j=ystart;j<h;j+=yoff)
		for(int k=xstart;k<w;k+=xoff){
			 /*
			  *DstT v = src[j*w+i*h*w+k]
			  *        -src[clamp(j+y,h)*w+clamp(i+z,d)*w*h+ clamp(k+x,w)];
			  */
			/*
			 *DstT v = *PITCH(src,spitch,j+i*h,k)
			 *        -*PITCH(src,spitch,clamp(j+y,h)+clamp(i+z,d)*h, clamp(k+x,w) );
			 */
			DstT v =  tex3D(texref<3,SrcT>::get(), k, j, i)
			       - tex3D(texref<3,SrcT>::get(), k+x,j+y,i+z);
				;
			v = ((k+x>=w) | (j+y>=h) | (i+z>=d)) ? 1E6 : v;
			diffs[k+j*w+i*w*h] = v*v;
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
	void filter_nlmean(cuv::tensor<T,dev_memory_space,row_major>& dst, const cuv::tensor<T,dev_memory_space,row_major,memory2d_tag>& constsrc, int search_radius, int filter_radius, float sigma, float dist_sigma, float step_size, bool threeDim, bool verbose){
		cuvAssert(!threeDim || constsrc.ndim()==3);

		bool d3 = constsrc.ndim()==3;
		unsigned int w = constsrc.shape()[d3?2:1], h=constsrc.shape()[d3?1:0], d=d3?constsrc.shape()[0]:1;
		const tensor<float,dev_memory_space,row_major,memory2d_tag> src(indices[index_range(0,d)][index_range(0,h)][index_range(0,w)], constsrc);

		if(!equal_shape(dst,src)){
			dst = cuv::tensor<T,dev_memory_space>(src.shape());
		}
		if(0){
			dst = 0.f;
			cuv::tensor<float,host_memory_space> kernel(2*filter_radius+1);
			for(int i=-filter_radius; i<=filter_radius;i++)
				kernel(i+filter_radius) = (float) exp(-i*i);
			kernel /= (float)cuv::sum(kernel);
			kernel = 1.f/kernel.size();
			setConvolutionKernel_horizontal(kernel);
			setConvolutionKernel_vertical(kernel);
			setConvolutionKernel_depth(kernel);
			cuv::tensor<float,dev_memory_space> tmp1(constsrc.shape());
			cuv::tensor<float,dev_memory_space> cpy(constsrc);
			/*convolutionRows   (dst,cpy,filter_radius);*/
			convolutionColumns(dst,cpy,filter_radius);
			/*convolutionDepth  (dst,cpy,filter_radius);*/
			return;
		}
		cuda_array<T,dev_memory_space> ca(src.shape()[1],src.shape()[2],src.shape()[0],1);
		ca.assign(src);

		typedef typename texref<3,T>::type textype;
		textype& tex = texref<3,T>::get();
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
		tex.normalized = false;
		tex.filterMode = cudaFilterModePoint;
		tex.addressMode[0] = cudaAddressModeClamp;
		tex.addressMode[1] = cudaAddressModeClamp;
		tex.addressMode[2] = cudaAddressModeClamp;
		cuvSafeCall(cudaBindTextureToArray(tex, ca.ptr(), channelDesc));

		cuv::tensor<float,dev_memory_space> weights(src.shape());
		cuv::tensor<float,dev_memory_space> diffs(src.shape());
		cuv::tensor<float,dev_memory_space> tmp1(src.shape());

		// prepare kernel
		cuv::tensor<float,host_memory_space> kernel(2*filter_radius+1);
		if(filter_radius!=0){
			kernel = 1.f/kernel.size();
			setConvolutionKernel_horizontal(kernel);
			setConvolutionKernel_vertical(kernel);
			setConvolutionKernel_depth(kernel);
		}

		dst     = (T)0.f;
		weights = (T)0.f;
		typedef float step_type;
		if(threeDim){
			dim3 blocks(divup(w,8),divup(h,8),divup(d,8));
			dim3 threads(8,8,8);
			int fw=(2*search_radius+1)*1.f/step_size;
			ProgressBar pb(fw*fw*fw);
			for(step_type i=-search_radius;i<=search_radius;i+=step_size){
				for(step_type j=-search_radius;j<=search_radius;j+=step_size){
					for(step_type k=-search_radius;k<=search_radius;k+=step_size){
						get_sqdiff<<<blocks,threads>>>(diffs.ptr(),src.ptr(),k,j,i,w,h,d,src.pitch());
						if(filter_radius==0){
							tmp1 = diffs;
						}else{
							convolutionRows   (tmp1,diffs,filter_radius);
							convolutionColumns(diffs,tmp1,filter_radius);
							convolutionDepth  (tmp1,diffs,filter_radius);
						}
						tmp1 /= -sigma*sigma;
						if(dist_sigma>0.f)
							tmp1 += -(i*i+j*j+k*k)/(dist_sigma*dist_sigma);
						cuv::apply_scalar_functor(tmp1, SF_EXP);
						weights += tmp1;
						mult_offset<false><<<blocks,threads>>>(dst.ptr(),tmp1.ptr(),src.ptr(),k,j,i,w,h,d,src.pitch());
						if(verbose)
							pb.inc();
					}
				}
			}
			if(verbose)
				pb.finish();
			dst /= weights;
		}else{
			dim3 blocks(divup(w,8),divup(h,8),1);
			dim3 threads(16,16,1);
			int fw=(2*search_radius+1)*1.f/step_size;
			ProgressBar pb(fw*fw);
			for(step_type k=-search_radius;k<=search_radius;k+=step_size){
				for(step_type j=-search_radius;j<=search_radius;j+=step_size){
					get_sqdiff2d<<<blocks,threads>>>(diffs.ptr(),src.ptr(),k,j,(step_type)0,w,h,d,src.pitch());
					if(filter_radius!=0){
						convolutionRows(tmp1,diffs,filter_radius);
						convolutionColumns(diffs,tmp1,filter_radius);
					}
					diffs /= -sigma*sigma;
					if(dist_sigma>0.f)
						diffs += -(j*j+k*k)/(dist_sigma*dist_sigma);
					cuv::apply_scalar_functor(diffs, SF_EXP);
					weights += diffs;
					mult_offset<true><<<blocks,threads>>>(dst.ptr(),diffs.ptr(),src.ptr(),k,j,(step_type)0,w,h,d,src.pitch());
					if(verbose)
						pb.inc();
				}
			}
			if(verbose)
				pb.finish();
			tensor<float,dev_memory_space> view_w(indices[0][index_range(0,h)][index_range(0,w)],weights);
			for(int l=0;l<d;l++){
				tensor<float,dev_memory_space> view_dst(indices[l][index_range(0,h)][index_range(0,w)],dst);
				view_dst/=view_w;
			}
		}
		cuvSafeCall(cudaUnbindTexture(tex));
	}

	/*template void filter_nlmean(cuv::tensor<float,dev_memory_space>& dst, const cuv::tensor<float,dev_memory_space>& src);*/
	template void filter_nlmean(cuv::tensor<float,dev_memory_space,row_major>& dst, const cuv::tensor<float,dev_memory_space,row_major,memory2d_tag>& src, int,int,float,float,float,bool,bool);
}
}
}
