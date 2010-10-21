#include "image_pyramid.hpp"

#define iDivUp(X,Y) (ceil((X)/(float)(Y)))
#define CB_TILE_W  16
#define CB_TILE_H  16
#define KERNEL_SIZE 5
#define HALF_KERNEL 2
#define NORM_FACTOR 0.00390625f // 1.0/(16^2)

texture<float,         2, cudaReadModeElementType> ip_float_tex; 
texture<unsigned char, 2, cudaReadModeElementType> ip_uc_tex; 

template<class T> struct texref{ };
template<> struct texref<float>{ 
	typedef texture<float, 2, cudaReadModeElementType> type;
	static type& get(){ return ip_float_tex; }; 
	__device__ float         operator()(float i, float j){return tex2D(ip_float_tex, i,j);} };
template<> struct texref<unsigned char>{ 
	typedef texture<unsigned char, 2, cudaReadModeElementType> type;
	static type& get(){ return ip_uc_tex; }; 
	__device__ unsigned char operator()(float i, float j){return tex2D(ip_uc_tex,i,j);} };


namespace cuv{
	//                         
	// Gaussian 5 x 5 kernel = [1, 4, 6, 4, 1]/16
	//
	template<class T>
	__global__
		void
		gaussian_pyramid_downsample_kernel(T* downLevel,
				size_t downLevelPitch,
				unsigned int downWidth, unsigned int downHeight)
		{
			// calculate normalized texture coordinates
			unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

			if(x < downWidth && y < downHeight) {
				float buf[KERNEL_SIZE];

				float u0 = (2.f * x) - HALF_KERNEL;
				float v0 = (2.f * y) - HALF_KERNEL;

				texref<T> tex;
				for(int i = 0; i < KERNEL_SIZE; i++) {
					buf[i] = 
						(    tex(u0    , v0 + i) + tex(u0 + 4, v0 + i)) + 
						4 * (tex(u0 + 1, v0 + i) + tex(u0 + 3, v0 + i)) +
						6 *  tex(u0 + 2, v0 + 2);
				}

				downLevel[y * downLevelPitch + x] = (buf[0] + buf[4] + 4*(buf[1] + buf[3]) + 6 * buf[2]) * NORM_FACTOR;
			}
		}


	template<class V,class S, class I>
		void gaussian_pyramid_downsample(
				dense_matrix<V,row_major,S,I>& dst,
				const cuda_array<V,S,I>& src){

			dim3 grid(iDivUp(dst.w(), CB_TILE_W), iDivUp(dst.h(), CB_TILE_H));
			dim3 threads(CB_TILE_W, CB_TILE_H);

			typedef typename texref<V>::type textype;
			textype& tex = texref<V>::get();
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<V>();
			tex.normalized = false;
			tex.filterMode = cudaFilterModePoint;
			tex.addressMode[0] = cudaAddressModeClamp;
			tex.addressMode[1] = cudaAddressModeClamp;
			cudaBindTextureToArray(tex, src.ptr(), channelDesc);
			checkCudaError("cudaBindTextureToArray");

			gaussian_pyramid_downsample_kernel<<<grid,threads>>>(dst.ptr(),
					dst.w(),
					dst.w(),
					dst.h());
			cuvSafeCall(cudaThreadSynchronize());

			cudaUnbindTexture(tex);
			checkCudaError("cudaUnbindTexture");
		}

	// explicit instantiation
	template void gaussian_pyramid_downsample(
			dense_matrix<float,row_major,dev_memory_space,unsigned int>& dst,
			const cuda_array<float,dev_memory_space,unsigned int>& src);
	template void gaussian_pyramid_downsample(
			dense_matrix<unsigned char,row_major,dev_memory_space,unsigned int>& dst,
			const cuda_array<unsigned char,dev_memory_space,unsigned int>& src);
}
