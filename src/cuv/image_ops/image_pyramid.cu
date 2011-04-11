#include <cuv/image_ops/image_pyramid.hpp>

#define iDivUp(X,Y) (ceil((X)/(float)(Y)))
#define CB_TILE_W  16
#define CB_TILE_H  16
#define KERNEL_SIZE 5
#define HALF_KERNEL 2
#define NORM_FACTOR 0.00390625f // 1.0/(16^2)

texture<float,         2, cudaReadModeElementType> ip_float_tex; 
texture<unsigned char, 2, cudaReadModeElementType> ip_uc_tex; 
texture<float4,        2, cudaReadModeElementType> ip_float4_tex; 
texture<uchar4,        2, cudaReadModeElementType> ip_uc4_tex; 

template<class T> struct texref{ };
template<> struct texref<float>{ 
	typedef texture<float, 2, cudaReadModeElementType> type;
	static type& get(){ return ip_float_tex; }; 
	__device__ float         operator()(float i, float j){return tex2D(ip_float_tex, i,j);} };
template<> struct texref<unsigned char>{ 
	typedef texture<unsigned char, 2, cudaReadModeElementType> type;
	static type& get(){ return ip_uc_tex; }; 
	__device__ unsigned char operator()(float i, float j){return tex2D(ip_uc_tex,i,j);} };
template<> struct texref<float4>{ 
	typedef texture<float4, 2, cudaReadModeElementType> type;
	static type& get(){ return ip_float4_tex; }; 
	__device__ float4        operator()(float i, float j){return tex2D(ip_float4_tex, i,j);} };
template<> struct texref<uchar4>{ 
	typedef texture<uchar4, 2, cudaReadModeElementType> type;
	static type& get(){ return ip_uc4_tex; }; 
	__device__ uchar4 operator()(float i, float j){return tex2D(ip_uc4_tex,i,j);} };



namespace cuv{
	template<class T> __device__ T plus4(const T& a, const T& b){ 
		T tmp = a;
		tmp.x += b.x;
		tmp.y += b.y;
		tmp.z += b.z;
		return tmp;
	}
	template<class T, class S> __device__ T mul4 (const S& s, const T& a){ 
		T tmp = a;
		tmp.x *= s;
		tmp.y *= s;
		tmp.z *= s;
		return tmp;
	}
	//                         
	// Gaussian 5 x 5 kernel = [1, 4, 6, 4, 1]/16
	//
	template<class T4, class T>
	__global__
		void
		gaussian_pyramid_downsample_kernel4val(T* downLevel,
				size_t downLevelPitch,
				unsigned int downWidth, unsigned int downHeight)
		{
			// calculate normalized texture coordinates
			unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

			T4 buf[KERNEL_SIZE];

			if(x < downWidth && y < downHeight) {

				float u0 = (2.f * x) - HALF_KERNEL;
				float v0 = (2.f * y) - HALF_KERNEL;

				texref<T4> tex;
				for(int i = 0; i < KERNEL_SIZE; i++) {
					T4 tmp;
					tmp = plus4(                   tex(u0    , v0 + i) , tex(u0 + 4, v0 + i));
					tmp = plus4(tmp, mul4(4, plus4(tex(u0 + 1, v0 + i) , tex(u0 + 3, v0 + i))));
					tmp = plus4(tmp, mul4(6,       tex(u0 + 2, v0 + 2)));
					buf[i] = tmp;
				}

				unsigned int pos = y*downLevelPitch + x;
				downLevel[pos + 0*downLevelPitch*downHeight] = (buf[0].x + buf[4].x + 4*(buf[1].x + buf[3].x) + 6 * buf[2].x) * NORM_FACTOR;
				downLevel[pos + 1*downLevelPitch*downHeight] = (buf[0].y + buf[4].y + 4*(buf[1].y + buf[3].y) + 6 * buf[2].y) * NORM_FACTOR;
				downLevel[pos + 2*downLevelPitch*downHeight] = (buf[0].z + buf[4].z + 4*(buf[1].z + buf[3].z) + 6 * buf[2].z) * NORM_FACTOR;
			}
		}
	//                         
	// Gaussian 5 x 5 kernel = [1, 4, 6, 4, 1]/16
	// inspired by http://sourceforge.net/projects/openvidia/files/CUDA%20Bayesian%20Optical%20Flow/
	// with bugfix...
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

	// Gaussian 5 x 5 kernel = [1, 4, 6, 4, 1]/16
	//
	template<class T>
	__global__
		void
		gaussian_kernel(T* dst,
				size_t dstPitch,
				unsigned int dstWidth, unsigned int dstHeight)
		{
			// calculate normalized texture coordinates
			unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

			if(x < dstWidth && y < dstHeight) {
				float buf[KERNEL_SIZE];

				float u0 = x - (float)HALF_KERNEL;
				float v0 = y - (float)HALF_KERNEL;

				texref<T> tex;
				for(int i = 0; i < KERNEL_SIZE; i++) {
					buf[i] = 
						(    tex(u0    , v0 + i) + tex(u0 + 4, v0 + i)) + 
						4 * (tex(u0 + 1, v0 + i) + tex(u0 + 3, v0 + i)) +
						6 *  tex(u0 + 2, v0 + 2);
				}

				dst[y * dstPitch + x] = (buf[0] + buf[4] + 4*(buf[1] + buf[3]) + 6 * buf[2]) * NORM_FACTOR;
			}
		}
	template<class T>
	__global__
		void
		gaussian_pyramid_upsample_kernel(T* upLevel,
				size_t upLevelPitch,
				unsigned int upWidth, unsigned int upHeight)
		{
			// calculate normalized texture coordinates
			unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

			if(x < upWidth && y < upHeight) {
				float u0 = (x/2.f);
				float v0 = (y/2.f);

				texref<T> tex;
				upLevel[y * upLevelPitch + x] = tex(u0,v0);
			}
		}


	template<class T> struct single_to_4{};
	template<>        struct single_to_4<float>        {typedef float4 type;};
	template<>        struct single_to_4<unsigned char>{typedef uchar4 type;};
	template<class V,class S, class I>
		void gaussian(
				tensor<V,S,row_major>& dst,
				const cuda_array<V,S,I>& src){
                        cuvAssert(dst.shape().size()==2);

			typedef typename texref<V>::type textype;
			textype& tex = texref<V>::get();
			tex.normalized = false;
			tex.filterMode = cudaFilterModePoint;
			tex.addressMode[0] = cudaAddressModeClamp;
			tex.addressMode[1] = cudaAddressModeClamp;

			dim3 grid,threads;
			grid = dim3 (iDivUp(dst.shape()[1], CB_TILE_W), iDivUp(dst.shape()[0], CB_TILE_H));
			threads = dim3 (CB_TILE_W, CB_TILE_H);
			cuvAssert(dst.shape()[1] == src.w());
			cuvAssert(dst.shape()[0] == src.h());
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<V>();
			cudaBindTextureToArray(tex, src.ptr(), channelDesc);
			checkCudaError("cudaBindTextureToArray");
			gaussian_kernel<<<grid,threads>>>(dst.ptr(),
					dst.shape()[1],
					dst.shape()[1],
					dst.shape()[0]);
			cuvSafeCall(cudaThreadSynchronize());
			cudaUnbindTexture(tex);
			checkCudaError("cudaUnbindTexture");

		}
	template<class V,class S, class I>
		void gaussian_pyramid_downsample(
				tensor<V,S,row_major>& dst,
				const cuda_array<V,S,I>& src,
				const unsigned int interleaved_channels){
                        cuvAssert(dst.shape().size()==2);


			typedef typename single_to_4<V>::type V4;
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<V>();
			cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc<V4>();

			typedef typename texref<V>::type textype;
			typedef typename texref<V4>::type textype4;

			textype& tex = texref<V>::get();
			tex.normalized = false;
			tex.filterMode = cudaFilterModePoint;
			tex.addressMode[0] = cudaAddressModeClamp;
			tex.addressMode[1] = cudaAddressModeClamp;
			textype4& tex4 = texref<V4>::get();
			tex4.normalized = false;
			tex4.filterMode = cudaFilterModeLinear;
			tex4.addressMode[0] = cudaAddressModeClamp;
			tex4.addressMode[1] = cudaAddressModeClamp;

			dim3 grid,threads;
			switch(interleaved_channels){
				case 1: // deals with a single channel
					grid = dim3 (iDivUp(dst.shape()[1], CB_TILE_W), iDivUp(dst.shape()[0], CB_TILE_H));
					threads = dim3 (CB_TILE_W, CB_TILE_H);
					cuvAssert(dst.shape()[1] < src.w());
					cuvAssert(dst.shape()[0] < src.h());
					cudaBindTextureToArray(tex, src.ptr(), channelDesc);
					checkCudaError("cudaBindTextureToArray");
					gaussian_pyramid_downsample_kernel<<<grid,threads>>>(dst.ptr(),
							dst.shape()[1],
							dst.shape()[1],
							dst.shape()[0]);
					cuvSafeCall(cudaThreadSynchronize());
					cudaUnbindTexture(tex);
					checkCudaError("cudaUnbindTexture");
					break;
				case 4: // deals with 4 interleaved channels (and writes to 3(!))
					cuvAssert(dst.shape()[1]   < src.w());
					cuvAssert(dst.shape()[0] / 3 < src.h()); 
					cuvAssert(dst.shape()[0] % 3 == 0); // three channels in destination (non-interleaved)
					cuvAssert(src.dim()==4);
					grid    = dim3(iDivUp(dst.shape()[1], CB_TILE_W), iDivUp(dst.shape()[0]/3, CB_TILE_H));
					threads = dim3(CB_TILE_W, CB_TILE_H);
					fill(dst, (V)0);
					cudaBindTextureToArray(tex4, src.ptr(), channelDesc4);
					checkCudaError("cudaBindTextureToArray");
					gaussian_pyramid_downsample_kernel4val<V4,V><<<grid,threads>>>(
							dst.ptr(),
							dst.shape()[1],
							dst.shape()[1],
							dst.shape()[0]/3);
					cuvSafeCall(cudaThreadSynchronize());
					cudaUnbindTexture(tex4);
					checkCudaError("cudaUnbindTexture");
					break;
				default:
					cuvAssert(false);
			}
			cuvSafeCall(cudaThreadSynchronize());

		}

	// Upsampling with hardware linear interpolation
	template<class V,class S, class I>
		void gaussian_pyramid_upsample(
				tensor<V,S,row_major>& dst,
				const cuda_array<V,S,I>& src){
                        cuvAssert(dst.shape().size()==2);
			cuvAssert(dst.shape()[1] > src.w());
			cuvAssert(dst.shape()[0] > src.h());

			dim3 grid(iDivUp(dst.shape()[1], CB_TILE_W), iDivUp(dst.shape()[0], CB_TILE_H));
			dim3 threads(CB_TILE_W, CB_TILE_H);

			typedef typename texref<V>::type textype;
			textype& tex = texref<V>::get();
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<V>();
			tex.normalized = false;
			tex.filterMode = cudaFilterModeLinear;
			tex.addressMode[0] = cudaAddressModeClamp;
			tex.addressMode[1] = cudaAddressModeClamp;
			cudaBindTextureToArray(tex, src.ptr(), channelDesc);
			checkCudaError("cudaBindTextureToArray");

			gaussian_pyramid_upsample_kernel<<<grid,threads>>>(dst.ptr(),
					dst.shape()[1],
					dst.shape()[1],
					dst.shape()[0]);
			cuvSafeCall(cudaThreadSynchronize());

			cudaUnbindTexture(tex);
			checkCudaError("cudaUnbindTexture");
		}


	template<class T>
	__device__
	T colordist(float u0, float v0, 
			    float u1, float v1,
			const float& offset, const unsigned int& dim){
		T d0 = (T) 0;
		texref<T> tex;
		for(unsigned int i=0;i<dim;i++){
			 float f = tex(u0,v0) - tex(u1,v1);
			 d0 += f*f;
			 v0 += offset;
			 v1 += offset;
		}
		return d0;
	}
	template<class T>
	struct summer{
		T mt;
		__device__ summer():mt(0){}
		__device__ void operator()(const T& t ){ mt+=t; }
	};
	template<class T>
	struct expsummer{
		T mt;
		__device__ expsummer():mt(0){}
		__device__ void operator()(const T& t ){ mt+=exp(-t); }
	};
	template<class T, class TDest>
	__global__
		void
		get_pixel_classes_kernel(TDest* dst,
				size_t dstPitch, unsigned int dstWidth, unsigned int dstHeight,
				float offset,
				float scale_fact)
		{
			// calculate normalized texture coordinates
			unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

			const float N = 1.f;
			if(x < dstWidth && y < dstHeight) {
				float u0 = (x/scale_fact);
				float v0 = (y/scale_fact);

				summer<T> sum;

				unsigned char arg_min_cd = 0;
				T min_cd = colordist<T>(u0,v0,u0+N,v0+N,offset,3u);
				sum(min_cd);

				T val    = colordist<T>(u0,v0,u0+N,v0-N,offset,3u);
				if(val<min_cd){ min_cd = val; arg_min_cd = 1; }
				sum(val);

				val      = colordist<T>(u0,v0,u0-N,v0+N,offset,3u);
				if(val<min_cd){ min_cd = val; arg_min_cd = 2; }
				sum(val);

				val      = colordist<T>(u0,v0,u0-N,v0-N,offset,3u);
				if(val<min_cd){ min_cd = val; arg_min_cd = 3; }
				sum(val);


				TDest tmp = make_uchar4( 
						arg_min_cd % 2 ? 255: 0,
						arg_min_cd > 1 ? 255: 0,0,
						max(0.f,min(255.f,sum.mt - 4*min_cd)) // for summer
						/*max(0.f,min(255.f,255.f * exp(-min_cd)/sum.mt)) // for expsummer*/
						);
				dst[y * dstPitch + x] = tmp;;
			}
		}


	// determine a number out of [0,3] for every pixel which should vary
	// smoothly and according to detail level in the image
	template<class VDest, class V, class S, class I>
		void get_pixel_classes(
			tensor<VDest,S,row_major>& dst,
			const cuda_array<V,S,I>&             src_smooth,
			float scale_fact
		){
                        cuvAssert(dst.shape().size()==2);
			dim3 grid(iDivUp(dst.shape()[1], CB_TILE_W), iDivUp(dst.shape()[0], CB_TILE_H));
			dim3 threads(CB_TILE_W, CB_TILE_H);

			typedef typename texref<V>::type textype;
			textype& tex = texref<V>::get();
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<V>();
			tex.normalized = false;
			tex.filterMode = cudaFilterModeLinear;
			tex.addressMode[0] = cudaAddressModeClamp;
			tex.addressMode[1] = cudaAddressModeClamp;
			cudaBindTextureToArray(tex, src_smooth.ptr(), channelDesc);
			checkCudaError("cudaBindTextureToArray");

			cuvAssert(src_smooth.h() % 3 == 0);
			cuvAssert(dst.shape()[1] % 4 == 0); // float4!
			float offset = src_smooth.h()/3;
			offset=0;
			get_pixel_classes_kernel<float><<<grid,threads>>>((uchar4*)dst.ptr(),
					dst.shape()[1]/4, dst.shape()[1]/4, dst.shape()[0],
					offset,
					scale_fact
					);
			cuvSafeCall(cudaThreadSynchronize());

			cudaUnbindTexture(tex);
			checkCudaError("cudaUnbindTexture");
		}

	// explicit instantiation
	template void gaussian(
			tensor<float,dev_memory_space,row_major>& dst,
			const cuda_array<float,dev_memory_space,unsigned int>& src);
	template void gaussian(
			tensor<unsigned char,dev_memory_space,row_major>& dst,
			const cuda_array<unsigned char,dev_memory_space,unsigned int>& src);

	template void gaussian_pyramid_downsample(
			tensor<float,dev_memory_space,row_major>& dst,
			const cuda_array<float,dev_memory_space,unsigned int>& src,
			const unsigned int);
	template void gaussian_pyramid_downsample(
			tensor<unsigned char,dev_memory_space,row_major>& dst,
			const cuda_array<unsigned char,dev_memory_space,unsigned int>& src,
			const unsigned int);
	template void gaussian_pyramid_upsample(
			tensor<float,dev_memory_space,row_major>& dst,
			const cuda_array<float,dev_memory_space,unsigned int>& src);
	template void gaussian_pyramid_upsample(
			tensor<unsigned char,dev_memory_space,row_major>& dst,
			const cuda_array<unsigned char,dev_memory_space,unsigned int>& src);

	template void get_pixel_classes(
			tensor<unsigned char,dev_memory_space,row_major>& dst,
			const cuda_array<unsigned char,dev_memory_space,unsigned int>& src,
			float scale_fact);
	template void get_pixel_classes(
			tensor<float,dev_memory_space,row_major>& dst,
			const cuda_array<float,dev_memory_space,unsigned int>& src,
			float scale_fact);
	template void get_pixel_classes(
			tensor<unsigned char,dev_memory_space,row_major>& dst,
			const cuda_array<float,dev_memory_space,unsigned int>& src,
			float scale_fact);
}
