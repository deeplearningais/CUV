/*
 * Original source from nvidia cuda SDK 2.0
 * Modified by S. James Lee (sjames@evl.uic.edi)
 * 2008.12.05
 * Further modified by Hannes Schulz
 */

#include <cuv/basics/tensor.hpp>
#include <cuv/basics/dense_matrix.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/libs/separable_conv/separable_convolution.hpp>

namespace cuv{

	namespace sep_conv{

		//24-bit multiplication is faster on G80,
		//but we must be sure to multiply integers
		//only within [-8M, 8M - 1] range
#define IMUL(a, b) __mul24(a, b)

		////////////////////////////////////////////////////////////////////////////////
		// Kernel configuration
		////////////////////////////////////////////////////////////////////////////////
#define MAX_KERNEL_RADIUS 10
#define      MAX_KERNEL_W (2 * MAX_KERNEL_RADIUS + 1)
		__device__ __constant__ float d_Kernel[MAX_KERNEL_W];

		// Assuming ROW_TILE_W, KERNEL_RADIUS_ALIGNED and dataW 
		// are multiples of coalescing granularity size,
		// all global memory operations are coalesced in convolutionRowGPU()
#define            ROW_TILE_W 128
#define KERNEL_RADIUS_ALIGNED 16

		// Assuming COLUMN_TILE_W and dataW are multiples
		// of coalescing granularity size, all global memory operations 
		// are coalesced in convolutionColumnGPU()
#define COLUMN_TILE_W 16
#define COLUMN_TILE_H 48


		////////////////////////////////////////////////////////////////////////////////
		// Row convolution filter
		////////////////////////////////////////////////////////////////////////////////
		template<int KERNEL_RADIUS, class SrcT, class DstT>
			__global__ void convolutionRowGPU(
					DstT       *d_Result,
					const SrcT *d_Data,
					int dataW,
					int dataH
					){
				//Data cache
				__shared__ SrcT data[KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS];

				//Current tile and apron limits, relative to row start
				const int         tileStart = IMUL(blockIdx.x, ROW_TILE_W);
				const int           tileEnd = tileStart + ROW_TILE_W - 1;
				const int        apronStart = tileStart - KERNEL_RADIUS;
				const int          apronEnd = tileEnd   + KERNEL_RADIUS;

				//Clamp tile and apron limits by image borders
				const int    tileEndClamped = min(tileEnd, dataW - 1);
				const int apronStartClamped = max(apronStart, 0);
				const int   apronEndClamped = min(apronEnd, dataW - 1);

				//Row start index in d_Data[]
				const int          rowStart = IMUL(blockIdx.y, dataW);

				//Aligned apron start. Assuming dataW and ROW_TILE_W are multiples 
				//of half-warp size, rowStart + apronStartAligned is also a 
				//multiple of half-warp size, thus having proper alignment 
				//for coalesced d_Data[] read.
				const int apronStartAligned = tileStart - KERNEL_RADIUS_ALIGNED;

				const int loadPos = apronStartAligned + threadIdx.x;
				//Set the entire data cache contents
				//Load global memory values, if indices are within the image borders,
				//or initialize with zeroes otherwise
				if(loadPos >= apronStart){
					const int smemPos = loadPos - apronStart;

					data[smemPos] = 
						((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ?
						d_Data[rowStart + loadPos] : 0;
				}


				//Ensure the completness of the loading stage
				//because results, emitted by each thread depend on the data,
				//loaded by another threads
				__syncthreads();
				const int writePos = tileStart + threadIdx.x;

				//Assuming dataW and ROW_TILE_W are multiples of half-warp size,
				//rowStart + tileStart is also a multiple of half-warp size,
				//thus having proper alignment for coalesced d_Result[] write.
				if(writePos <= tileEndClamped){
					const int smemPos = writePos - apronStart;
					DstT sum = 0;

					for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
						sum += data[smemPos + k] * d_Kernel[KERNEL_RADIUS - k];

					d_Result[rowStart + writePos] = sum;
				}
			}



		////////////////////////////////////////////////////////////////////////////////
		// Column convolution filter
		////////////////////////////////////////////////////////////////////////////////
		template<int KERNEL_RADIUS, class SrcT, class DstT>
			__global__ void convolutionColumnGPU(
					DstT       *d_Result,
					const SrcT *d_Data,
					int dataW,
					int dataH,
					int smemStride,
					int gmemStride
					){
				//Data cache
				__shared__ SrcT data[COLUMN_TILE_W * (KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)];

				//Current tile and apron limits, in rows
				const int         tileStart = IMUL(blockIdx.y, COLUMN_TILE_H);
				const int           tileEnd = tileStart + COLUMN_TILE_H - 1;
				const int        apronStart = tileStart - KERNEL_RADIUS;
				const int          apronEnd = tileEnd   + KERNEL_RADIUS;

				//Clamp tile and apron limits by image borders
				const int    tileEndClamped = min(tileEnd, dataH - 1);
				const int apronStartClamped = max(apronStart, 0);
				const int   apronEndClamped = min(apronEnd, dataH - 1);

				//Current column index
				const int       columnStart = IMUL(blockIdx.x, COLUMN_TILE_W) + threadIdx.x;

				//Shared and global memory indices for current column
				int smemPos = IMUL(threadIdx.y, COLUMN_TILE_W) + threadIdx.x;
				int gmemPos = IMUL(apronStart + threadIdx.y, dataW) + columnStart;

				//Cycle through the entire data cache
				//Load global memory values, if indices are within the image borders,
				//or initialize with zero otherwise
				for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y){
					data[smemPos] = 
						((y >= apronStartClamped) && (y <= apronEndClamped)) ? 
						d_Data[gmemPos] : 0;
					smemPos += smemStride;
					gmemPos += gmemStride;
				}

				//Ensure the completness of the loading stage
				//because results, emitted by each thread depend on the data, 
				//loaded by another threads
				__syncthreads();

				//Shared and global memory indices for current column
				smemPos = IMUL(threadIdx.y + KERNEL_RADIUS, COLUMN_TILE_W) + threadIdx.x;
				gmemPos = IMUL(tileStart + threadIdx.y , dataW) + columnStart;

				//Cycle through the tile body, clamped by image borders
				//Calculate and output the results
				for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y){
					DstT sum = 0;

					for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
						sum += 
							data[smemPos + IMUL(k, COLUMN_TILE_W)] *
							d_Kernel[KERNEL_RADIUS - k];

					d_Result[gmemPos] = sum;
					smemPos += smemStride;
					gmemPos += gmemStride;
				}
			}

		int iDivUp(int a, int b){
			return (a % b != 0) ? (a / b + 1) : (a / b);
		}


#define V(X) #X << " : "<< (X)<<"  "
		template<int radius, class DstV, class SrcV, class I>
		void convolve(dense_matrix<DstV,row_major,dev_memory_space,I>& dst,
				     const dense_matrix<SrcV,row_major,dev_memory_space,I>& src, int dir=2){

			int dw = src.w();
			int dh = src.h();
			dim3 blockGridRows(iDivUp(dw, ROW_TILE_W), dh);
			dim3 threadBlockRows(KERNEL_RADIUS_ALIGNED + ROW_TILE_W + radius);	// 16 128 8
			dim3 blockGridColumns(iDivUp(dw, COLUMN_TILE_W), iDivUp(dh, COLUMN_TILE_H));
			dim3 threadBlockColumns(COLUMN_TILE_W, 8);
			
			if(dir==2){
				dense_matrix<DstV,row_major,dev_memory_space,I> intermed(dst.h(),dst.w());
				convolutionRowGPU<radius><<<blockGridRows, threadBlockRows>>>( intermed.ptr(), src.ptr(), src.w(), src.h());
				convolutionColumnGPU<radius><<<blockGridColumns, threadBlockColumns>>>( dst.ptr(), intermed.ptr(), intermed.w(), intermed.h(), COLUMN_TILE_W * threadBlockColumns.y, intermed.w() * threadBlockColumns.y);
			}
			else if(dir==0){
				convolutionRowGPU<radius><<<blockGridRows, threadBlockRows>>>( dst.ptr(), src.ptr(), src.w(), src.h());
			}else if(dir==1){
				convolutionColumnGPU<radius><<<blockGridColumns, threadBlockColumns>>>( dst.ptr(), src.ptr(), src.w(), src.h(), COLUMN_TILE_W * threadBlockColumns.y, src.w() * threadBlockColumns.y);
			}
			cuvSafeCall(cudaThreadSynchronize());
			safeThreadSync();
		}


		template<class DstV, class SrcV, class M, class I>
		void radius_dispatch(const unsigned int& radius,dense_matrix<DstV,row_major,M,I>& dst,
				     const dense_matrix<SrcV,row_major,M,I>& src,int dir=2){
			switch(radius){
				case 1: convolve<1>(dst,src,dir); break;
				case 2: convolve<2>(dst,src,dir); break;
				case 3: convolve<3>(dst,src,dir); break;
				case 4: convolve<4>(dst,src,dir); break;
				case 5: convolve<5>(dst,src,dir); break;
				case 6: convolve<6>(dst,src,dir); break;
				case 7: convolve<7>(dst,src,dir); break;
				case 8: convolve<8>(dst,src,dir); break;
				default: cuvAssert(false);
			}
		}
		template<class DstV, class SrcV, class M, class I>
		boost::ptr_vector<dense_matrix<DstV,row_major,M,I> >
		convolve( const dense_matrix<SrcV,row_major,M,I>& src,
			  const unsigned int& radius,
			  const separable_filter& filt ){

			typedef dense_matrix<DstV,row_major,M,I> result_type;
			cuvAssert(radius <= MAX_KERNEL_RADIUS);
			boost::ptr_vector<result_type> res;

			if(filt      == SP_GAUSS){
				const int kernel_w = 2*radius+1;
				cuv::tensor<float, host_memory_space> kernel(kernel_w);
				for(int i = 0; i < kernel_w; i++){
					float dist = (float)(i - radius) / (float)radius;
					kernel[i]=expf(- dist * dist / 2);
				}
				kernel /= cuv::sum(kernel);
				cuvSafeCall( cudaMemcpyToSymbol(d_Kernel, kernel.ptr(), kernel.memsize()) );
				res.push_back(new result_type(src.h(),src.w()));
				radius_dispatch(radius,res.back(),src,2);
			}else if(filt == SP_SOBEL){
				boost::ptr_vector<result_type> intermed;
				intermed = convolve<DstV>(src,radius,SP_GAUSS);

				const int kernel_w = 3;
				cuv::tensor<float, host_memory_space> kernel(kernel_w);
				kernel[0]=-0.5;
				kernel[1]= 0;
				kernel[2]= 0.5;
				cuvSafeCall( cudaMemcpyToSymbol(d_Kernel, kernel.ptr(), kernel.memsize()) );

				res.push_back(new result_type(src.h(),src.w()));
				radius_dispatch(1,res.back(),intermed.front(),0);

				res.push_back(new result_type(src.h(),src.w()));
				radius_dispatch(1,res.back(),intermed.front(),1);
			}
			return res;
		}
		
		// instantiations
#define INST(DSTV, SRCV,M, I) \
		template boost::ptr_vector<dense_matrix<DSTV,row_major,M,I> > \
		convolve<DSTV,SRCV,M,I>( const dense_matrix<SRCV,row_major,M,I>&, \
				                      const unsigned int&,                     \
				                      const separable_filter&);
		INST(float,float,dev_memory_space,unsigned int);
	} // namespace separable convolution
} // namespace cuv
