/*
 * Original source from nvidia cuda SDK 4.0
 */

#include <cuv/basics/tensor.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/libs/separable_conv/separable_convolution.hpp>

namespace cuv{

	namespace sep_conv{


#define PITCH(PTR,PITCH,Y,X) ((typeof(PTR))((char*)PTR + PITCH*Y) + X)
#define MAX_KERNEL_RADIUS 8
#define      MAX_KERNEL_W (2 * MAX_KERNEL_RADIUS + 1)
		__device__ __constant__ float c_Kernel[MAX_KERNEL_W];

		////////////////////////////////////////////////////////////////////////////////
		// Row convolution filter
		////////////////////////////////////////////////////////////////////////////////
#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

		template<int KERNEL_RADIUS, class SrcT, class DstT>
		__global__ void convolutionRowGPU(
				SrcT *d_Dst,
				DstT *d_Src,
				int imageW,
				int imageH,
				int dpitch,
				int spitch 
				){
			__shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

			//Offset to the left halo edge
			const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
			const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

			d_Src = PITCH(d_Src, spitch, baseY, baseX);
			d_Dst = PITCH(d_Dst, dpitch, baseY, baseX);

			//Load main data
#pragma unroll
			for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
				s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X)  ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

			//Load left halo
#pragma unroll
			for(int i = 0; i < ROWS_HALO_STEPS; i++)
				s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X ) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

			//Load right halo
#pragma unroll
			for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
				s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

			//Compute and store results
			__syncthreads();
#pragma unroll
			for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){
				float sum = 0;

#pragma unroll
				for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
					sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];

				d_Dst[i * ROWS_BLOCKDIM_X] = sum;
			}
		}

		////////////////////////////////////////////////////////////////////////////////
		// Column convolution filter
		////////////////////////////////////////////////////////////////////////////////
#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 1

		template<int KERNEL_RADIUS, class SrcT, class DstT>
		__global__ void convolutionColumnGPU(
				SrcT *d_Dst,
				DstT *d_Src,
				int imageW,
				int imageH,
				int dpitch,
				int spitch
				){
			__shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

			//Offset to the upper halo edge
			const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
			const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
			d_Src = PITCH(d_Src, spitch, baseY, baseX);
			d_Dst = PITCH(d_Dst, dpitch, baseY, baseX);

			//Main data
#pragma unroll
			for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
				s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (imageH-baseY > i * COLUMNS_BLOCKDIM_Y) ? *PITCH(d_Src, spitch, i*COLUMNS_BLOCKDIM_Y,0) : 0;

			//Upper halo
#pragma unroll
			for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
				s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? *PITCH(d_Src,spitch,i*COLUMNS_BLOCKDIM_Y,0) : 0;

			//Lower halo
#pragma unroll
			for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
				s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? *PITCH(d_Src,spitch,i*COLUMNS_BLOCKDIM_Y,0) : 0;

			//Compute and store results
			__syncthreads();
#pragma unroll
			for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
				float sum = 0;
#pragma unroll
				for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
					sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
				if(imageH - baseY > i * COLUMNS_BLOCKDIM_Y)
					*PITCH(d_Dst,dpitch,i*COLUMNS_BLOCKDIM_Y,0) = sum;
			}
		}


		int iDivUp(int a, int b){  	  	 
			return (a % b != 0) ? (a / b + 1) : (a / b);  	  	 
		} 
#define V(X) #X << " : "<< (X)<<"  "
		template<int radius, class DstV, class SrcV, class A>
		void convolve_call_kernel(tensor<DstV,dev_memory_space,row_major,A>& dst,
				     const tensor<SrcV,dev_memory_space,row_major,A>& src, int dir){

			int dw = src.shape()[1];
			int dh = src.shape()[0];
			
			if(dir==0){
				dim3 blocks(iDivUp(dw , (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X)), iDivUp(dh , ROWS_BLOCKDIM_Y));
				dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
				convolutionRowGPU<radius><<<blocks, threads>>>( dst.ptr(), src.ptr(), src.shape()[1], src.shape()[0],dst.pitch(),src.pitch());
			}else if(dir==1){
				dim3 blocks(iDivUp(dw , COLUMNS_BLOCKDIM_X), iDivUp(dh , (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)));
				dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
				convolutionColumnGPU<radius><<<blocks, threads>>>( dst.ptr(), src.ptr(), src.shape()[1], src.shape()[0], dst.pitch(), src.pitch());
			}
			cuvSafeCall(cudaThreadSynchronize());
		}


		template<class DstV, class SrcV, class M, class A>
		void radius_dispatch(const unsigned int& radius,tensor<DstV,M,row_major,A>& dst,
				     const tensor<SrcV,M,row_major,A>& src,int dir){
			switch(radius){
				case 1: convolve_call_kernel<1>(dst,src,dir); break;
				case 2: convolve_call_kernel<2>(dst,src,dir); break;
				case 3: convolve_call_kernel<3>(dst,src,dir); break;
				case 4: convolve_call_kernel<4>(dst,src,dir); break;
				case 5: convolve_call_kernel<5>(dst,src,dir); break;
				case 6: convolve_call_kernel<6>(dst,src,dir); break;
				case 7: convolve_call_kernel<7>(dst,src,dir); break;
				case 8: convolve_call_kernel<8>(dst,src,dir); break;
				default: cuvAssert(false);
			}
		}
		template<class DstV, class SrcV, class M, class A>
		void
		convolve(       tensor<DstV,M,row_major, A>& dst,
			  const tensor<SrcV,M,row_major, A>& src,
			  const unsigned int&   filter_radius,
			  const separable_filter& filt, int axis, 
			  const float& param ){

			typedef tensor<DstV,M,row_major,A> result_type;
			typedef tensor<SrcV,M,row_major,A>    src_type;
			cuvAssert(filter_radius <= MAX_KERNEL_RADIUS);
                        cuvAssert(src.ndim()==2 || src.ndim()==3);

			if(!equal_shape(dst,src)){
				dst = result_type(src.shape());
			}

			if(src.ndim()==3){
				const std::vector<typename src_type::index_type>& s = src.shape();
				for(unsigned int i=0;i<s[0];i++){
					src_type    sview(indices[i][index_range(0,s[1])][index_range(0,s[2])], src);
					result_type dview(indices[i][index_range(0,s[1])][index_range(0,s[2])], dst);
					convolve(dview,sview,filter_radius,filt,axis,param);
				}
				return;
			}

			if(filt == SP_GAUSS){
				const int kernel_w = 2*filter_radius+1;
				cuv::tensor<float, host_memory_space> kernel(kernel_w);
				for(int i = 0; i < kernel_w; i++){
					float dist = (float)(i - (int)filter_radius);
					kernel[i]  = expf(- dist * dist / (2*param*param));
				}
				kernel /= cuv::sum(kernel);
				cuvSafeCall( cudaMemcpyToSymbol(c_Kernel, kernel.ptr(), kernel.memsize()) );
				result_type tmp(extents[src.shape()[0]][src.shape()[1]]);
				radius_dispatch(filter_radius,tmp,src,0);
				radius_dispatch(filter_radius,dst,tmp,1);
			}else if(filt == SP_CENTERED_DERIVATIVE){
				cuvAssert(axis==0 || axis==1);
				cuv::tensor<float, host_memory_space> kernel(3);
				kernel[0]=-0.5;
				kernel[1]= 0;
				kernel[2]= 0.5;
				cuvSafeCall( cudaMemcpyToSymbol(c_Kernel, kernel.ptr(), kernel.memsize()) );
				radius_dispatch(1,dst,src,axis);
			}else if(filt == SP_BOX){
				const int kernel_w = 2*filter_radius+1;
				cuv::tensor<float, host_memory_space> kernel(kernel_w);
				cuv::fill(kernel, 1.f);
				kernel /= (float) kernel_w;
				cuvSafeCall( cudaMemcpyToSymbol(c_Kernel, kernel.ptr(), kernel.memsize()) );
				result_type tmp(extents[src.shape()[0]][src.shape()[1]]);
				radius_dispatch(filter_radius,tmp,src,0);
				radius_dispatch(filter_radius,dst,tmp,1);
			}
		}
		
		// instantiations
#define INST(DSTV, SRCV,M,A) \
		template void \
		convolve<DSTV,SRCV,M>( tensor<DSTV,M,row_major, A>&, \
				const tensor<SRCV,M,row_major, A>&, \
				const unsigned int&,                     \
				const separable_filter&, int axis, \
				const float&);
		INST(float,float,dev_memory_space,linear_memory_tag);
		INST(float,float,dev_memory_space,memory2d_tag);
	} // namespace separable convolution
} // namespace cuv
