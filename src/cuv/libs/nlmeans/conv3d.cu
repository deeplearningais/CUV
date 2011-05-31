/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation and
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution
 * of this software and related documentation without an express license
 * agreement from NVIDIA Corporation is strictly prohibited.
 *
 */


#include <cuv/basics/tensor.hpp>
#include <cuv/tools/cuv_general.hpp>
/*#include <math_functions.h>*/
// #include <vector>

namespace cuv{
	namespace libs{
		namespace nlmeans{


			////////////////////////////////////////////////////////////////////////////////
			// Convolution kernel storage
			////////////////////////////////////////////////////////////////////////////////
			__constant__ float c_Kernel_h[100];
			__constant__ float c_Kernel_v[100];
			__constant__ float c_Kernel_d[100];

			void setConvolutionKernel_horizontal(const cuv::tensor<float,host_memory_space>&src){
				cudaMemcpyToSymbol(c_Kernel_h, src.ptr(), src.size() * sizeof(float));
			}
			void setConvolutionKernel_vertical(const cuv::tensor<float,host_memory_space>&src){
				cudaMemcpyToSymbol(c_Kernel_v, src.ptr(), src.size() * sizeof(float));
			}
			void setConvolutionKernel_depth(const cuv::tensor<float,host_memory_space>&src){
				cudaMemcpyToSymbol(c_Kernel_d, src.ptr(), src.size() * sizeof(float));
			}

			////////////////////////////////////////////////////////////////////////////////
			// Constants
			////////////////////////////////////////////////////////////////////////////////
#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 16
#define   ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 3
#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 16
#define   COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 3
#define   DEPTH_BLOCKDIM_Y 16
#define   DEPTH_BLOCKDIM_Z 16
#define   DEPTH_RESULT_STEPS 4
#define   DEPTH_HALO_STEPS 3



			////////////////////////////////////////////////////////////////////////////////
			// Row convolution filter
			////////////////////////////////////////////////////////////////////////////////
			__global__ void convolutionRowsKernel(
					float *d_Dst,
					const float *d_Src,
					int imageW,
					int imageH,
					int imageD,
					int kernel_radius
					){
				__shared__ float s_Data[ROWS_BLOCKDIM_Y]
					[(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

				//Offset to the left halo edge
				int n_blocks_per_row = imageW/(ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X);
				int basez = floor(float(blockIdx.x)/n_blocks_per_row);

				int blockx = blockIdx.x - basez*n_blocks_per_row;
				// const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) *
				// ROWS_BLOCKDIM_X + threadIdx.x;
				const int baseX = (blockx * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) *
					ROWS_BLOCKDIM_X + threadIdx.x;
				const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

				d_Src += basez*imageW*imageH + baseY * imageW + baseX;
				d_Dst += basez*imageW*imageH + baseY * imageW + baseX;

				//Main data
#pragma unroll
				for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
					s_Data[threadIdx.y]
						[threadIdx.x + i * ROWS_BLOCKDIM_X]
						= d_Src[i * ROWS_BLOCKDIM_X];

				//Left halo
				for(int i = 0; i < ROWS_HALO_STEPS; i++){
					s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
						(baseX >= -i * ROWS_BLOCKDIM_X ) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
				}

				//Right halo
				for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS;
						i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++){
					s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
						(imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
				}

				//Compute and store results
				__syncthreads();
				// #pragma unroll
				for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){
					float sum = 0;

#pragma unroll
					for(int j = -kernel_radius; j <= kernel_radius; j++)
						sum += c_Kernel_h[kernel_radius - j] *
							s_Data    [threadIdx.y]
							[threadIdx.x + i * ROWS_BLOCKDIM_X + j];
					d_Dst[i * ROWS_BLOCKDIM_X] = sum;
				}
			}

			void convolutionRows(
					cuv::tensor<float,dev_memory_space> &d_Dst,
					const cuv::tensor<float,dev_memory_space> &d_Src,
					int kernel_radius
					){
				int imageW = d_Src.shape()[2];
				int imageH = d_Src.shape()[1];
				int imageD = d_Src.shape()[0];
				cuvAssert( ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= kernel_radius );
				//There is a rational division of the image into blocks
				cuvAssert( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
				cuvAssert( imageH % ROWS_BLOCKDIM_Y == 0 );

				dim3 blocks(imageD*(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X)),
						imageH / ROWS_BLOCKDIM_Y);
				dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
				convolutionRowsKernel<<<blocks, threads>>>(
						// &d_Dst[i*imageH*imageW],
						// &d_Src[i*imageH*imageW],
						d_Dst.ptr(),
						d_Src.ptr(),
						imageW,
						imageH,
						imageD,
						kernel_radius
						);
				cuvSafeCall(cudaThreadSynchronize());
			}



			////////////////////////////////////////////////////////////////////////////////
			// Column convolution filter
			////////////////////////////////////////////////////////////////////////////////
			__global__ void convolutionColumnsKernel(
					float *d_Dst,
					const float *d_Src,
					int imageW,
					int imageH,
					int pitch,
					int kernel_radius
					){
				__shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

				int n_blocks_per_column = imageH/(COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y);
				int basez = floor(float(blockIdx.y)/n_blocks_per_column);
				int blocky = blockIdx.y - basez*n_blocks_per_column;

				//Offset to the upper halo edge
				const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
				const int baseY = (blocky * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
				d_Src += basez*imageH*imageW + baseY * imageH + baseX;
				d_Dst += basez*imageH*imageW + baseY * imageH + baseX;

				//Main data
#pragma unroll
				for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
					s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];

				//Upper halo
				for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
					s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
						(baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

				//Lower halo
				for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
					s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
						(imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

				//Compute and store results
				__syncthreads();
				// #pragma unroll
				for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
					float sum = 0;
#pragma unroll
					for(int j = -kernel_radius; j <= kernel_radius; j++)
						sum += c_Kernel_v[kernel_radius - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];

					d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
				}
			}

			void convolutionColumns(
					cuv::tensor<float,dev_memory_space> & d_Dst,
					const cuv::tensor<float,dev_memory_space> & d_Src,
					int kernel_radius
					){

				int imageW = d_Src.shape()[2];
				int imageH = d_Src.shape()[1];
				int imageD = d_Src.shape()[0];
				cuvAssert( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= kernel_radius );
				cuvAssert( imageW % COLUMNS_BLOCKDIM_X == 0 );
				cuvAssert( imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );

				dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageD * imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
				dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

				convolutionColumnsKernel<<<blocks, threads>>>(
						d_Dst.ptr(),
						d_Src.ptr(),
						imageW,
						imageH,
						imageW,
						kernel_radius
						);
				cuvSafeCall(cudaThreadSynchronize());
			}

			////////////////////////////////////////////////////////////////////////////////
			// Depth convolution filter - Really naive implementation
			////////////////////////////////////////////////////////////////////////////////
			__global__ void convolutionDepthKernel(
					float *d_Dst,
					const float *d_Src,
					int imageW,
					int imageH,
					int imageD,
					int kernel_radius
					){
				__shared__ float s_Data[DEPTH_BLOCKDIM_Y]
					[(DEPTH_RESULT_STEPS + 2 * DEPTH_HALO_STEPS) * DEPTH_BLOCKDIM_Z];

				//Offset to the left halo edge
				int n_blocks_per_depth = imageD / (DEPTH_RESULT_STEPS * DEPTH_BLOCKDIM_Z);
				int basex = floor(float(blockIdx.x)/n_blocks_per_depth);

				int blockz = blockIdx.x - basex*n_blocks_per_depth;

				const int baseZ = (blockz * DEPTH_RESULT_STEPS - DEPTH_HALO_STEPS)*DEPTH_BLOCKDIM_Z +
					threadIdx.x;
				const int baseY = blockIdx.y * DEPTH_BLOCKDIM_Y + threadIdx.y;

				//Put the pointers to the beginning of the data
				d_Src += baseZ * imageW * imageH + baseY * imageW + basex;
				d_Dst += baseZ * imageW * imageH + baseY * imageW + basex;

				// //Main data
#pragma unroll
				for(int i = DEPTH_HALO_STEPS; i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS; i++)
					s_Data[threadIdx.y]
						[threadIdx.x + i * DEPTH_BLOCKDIM_Z]
						= d_Src[i * DEPTH_BLOCKDIM_Z * imageH * imageW];

				//Left halo
				for(int i = 0; i < DEPTH_HALO_STEPS; i++){
					s_Data[threadIdx.y][threadIdx.x + i * DEPTH_BLOCKDIM_Z] =
						(baseZ >= -i * DEPTH_BLOCKDIM_Z ) ?
						d_Src[i * DEPTH_BLOCKDIM_Z * imageH * imageW] : 0;
				}

				// Right halo
				for(int i = DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS;
						i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS + DEPTH_HALO_STEPS; i++){
					s_Data[threadIdx.y][threadIdx.x + i * DEPTH_BLOCKDIM_Z] =
						(imageD - baseZ > i * DEPTH_BLOCKDIM_Z ) ?
						d_Src[i * DEPTH_BLOCKDIM_Z * imageH * imageW] : 0;
				}

				// //Compute and store results
				__syncthreads();
				/*#pragma unroll*/
				for(int i = DEPTH_HALO_STEPS; i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS; i++){
					float sum = 0;
#pragma unroll
					for(int j = -kernel_radius; j <= kernel_radius; j++)
						sum += c_Kernel_d[kernel_radius - j] *
							s_Data    [threadIdx.y]
							[threadIdx.x + i * DEPTH_BLOCKDIM_Z + j];
					if(baseZ*imageW*imageH+baseY*imageW+basex+i*DEPTH_BLOCKDIM_Z*imageH*imageW < imageD*imageW*imageH)
						d_Dst[i * DEPTH_BLOCKDIM_Z * imageH * imageW] = sum;
				}
			}

			void convolutionDepth(
					cuv::tensor<float,dev_memory_space>& d_Dst,
					const cuv::tensor<float,dev_memory_space>& d_Src,
					int kernel_radius
					){
				int imageW = d_Src.shape()[2];
				int imageH = d_Src.shape()[1];
				int imageD = d_Src.shape()[0];

				cuvAssert( DEPTH_BLOCKDIM_Z * DEPTH_HALO_STEPS >= kernel_radius );
				//There is a rational division of the image into blocks
				cuvAssert( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
				cuvAssert( imageH % ROWS_BLOCKDIM_Y == 0 );

				dim3 blocks(imageW*imageD / (DEPTH_RESULT_STEPS * DEPTH_BLOCKDIM_Z),
						imageH / DEPTH_BLOCKDIM_Y);
				dim3 threads(DEPTH_BLOCKDIM_Z, DEPTH_BLOCKDIM_Y);

				// for(int x = 0; x < imageW; x++)
				convolutionDepthKernel<<<blocks, threads>>>(
						d_Dst.ptr(),
						d_Src.ptr(),
						imageW,
						imageH,
						imageD,
						kernel_radius
						);
				cuvSafeCall(cudaThreadSynchronize());
			}



			////////////////////////////////////////////////////////////////////////////////
			// Computes the higher eigenvalue of the hessian
			////////////////////////////////////////////////////////////////////////////////
			__device__ float computeDeterminant
				(float e00, float e01, float e02,
				 float e10, float e11, float e12,
				 float e20, float e21, float e22)
				{
					return e00*e11*e22-e00*e12*e21+e10*e21*e02-e10*e01*e22+e20*e01*e12-e20*e11*e02;
				}


			__global__ void hessianKernel
				(
				 float *d_output,
				 const float *d_gxx,
				 const float *d_gxy,
				 const float *d_gxz,
				 const float *d_gyy,
				 const float *d_gyz,
				 const float *d_gzz,
				 float sigma,
				 int imageW,
				 int imageH,
				 int imageD
				)
				{
					int n_blocks_per_width = imageW/blockDim.x;
					int z = (int)ceilf(blockIdx.x/n_blocks_per_width);
					int y = blockIdx.y*blockDim.y + threadIdx.y;
					int x = (blockIdx.x - z*n_blocks_per_width)*blockDim.x + threadIdx.x;
					int i = z*imageW*imageH + y*imageW + x;

					// // //Brute force eigen-values computation
					float a0, b0, c0, e0, f0, k0;
					a0 = -d_gxx[i]; b0 = -d_gxy[i]; c0 = -d_gxz[i];
					e0 = -d_gyy[i]; f0 = -d_gyz[i]; k0 = -d_gzz[i];


					// http://en.wikipedia.org/wiki/Eigenvalue_algorithm
					//Oliver K. Smith: Eigenvalues of a symmetric 3 × 3 matrix. Commun. ACM 4(4): 168 (1961)
					float m = (a0+e0+k0)/3;
					float q = computeDeterminant
						(a0-m, b0, c0, b0, e0-m, f0, c0, f0, k0-m)/2;
					float p = (a0-m)*(a0-m) + b0*b0 + c0*c0 + b0*b0 + (e0-m)*(e0-m) +
						f0*f0 + c0*c0 + f0*f0 + (k0-m)*(k0-m);
					p = p / 6;
					float phi = 1.f/3.f*atan(sqrt(p*p*p-q*q)/q);
					if(phi<0)
						phi=phi+3.14159f/3;

					float eig1 = m + 2*sqrt(p)*cos(phi);
					float eig2 = m - sqrt(p)*(cos(phi) + sqrt(3.0f)*sin(phi));
					float eig3 = m - sqrt(p)*(cos(phi) - sqrt(3.0f)*sin(phi));

					if( (eig1 > eig2) & (eig1 > eig3))
						d_output[i] = eig1*sigma*sigma;
					if( (eig2 > eig1) & (eig2 > eig3))
						d_output[i] = eig2*sigma*sigma;
					if( (eig3 > eig2) & (eig3 > eig1))
						d_output[i] = eig3*sigma*sigma;
				}



			void hessian
				(
				 cuv::tensor<float,dev_memory_space>& d_output,
				 const cuv::tensor<float,dev_memory_space>& d_gxx,
				 const cuv::tensor<float,dev_memory_space>& d_gxy,
				 const cuv::tensor<float,dev_memory_space>& d_gxz,
				 const cuv::tensor<float,dev_memory_space>& d_gyy,
				 const cuv::tensor<float,dev_memory_space>& d_gyz,
				 const cuv::tensor<float,dev_memory_space>& d_gzz,
				 float sigma
				)
				{
					int imageW = d_gxx.shape()[2];
					int imageH = d_gxx.shape()[1];
					int imageD = d_gxx.shape()[0];
					dim3 grid (imageD*imageW/ROWS_BLOCKDIM_X,imageH/ROWS_BLOCKDIM_Y);
					dim3 block(ROWS_BLOCKDIM_X,ROWS_BLOCKDIM_Y);
					hessianKernel<<<grid, block>>>( d_output.ptr(), d_gxx.ptr(), d_gxy.ptr(), d_gxz.ptr(),
							d_gyy.ptr(), d_gyz.ptr(), d_gzz.ptr(), sigma, imageW, imageH, imageD );
					cuvSafeCall(cudaThreadSynchronize());
				}



			/*********************************************************************************
			 ** hessian con orientacion
			 ********************************************************************************/
			__global__ void hessianKernelO
				(
				 float *d_output,
				 float *d_output_theta,
				 float *d_output_phi,
				 const float *d_gxx,
				 const float *d_gxy,
				 const float *d_gxz,
				 const float *d_gyy,
				 const float *d_gyz,
				 const float *d_gzz,
				 float sigma,
				 int imageW,
				 int imageH,
				 int imageD
				)
				{
					int n_blocks_per_width = imageW/blockDim.x;
					int z = (int)ceilf(blockIdx.x/n_blocks_per_width);
					int y = blockIdx.y*blockDim.y + threadIdx.y;
					int x = (blockIdx.x - z*n_blocks_per_width)*blockDim.x + threadIdx.x;
					int i = z*imageW*imageH + y*imageW + x;

					// // //Brute force eigen-values computation
					// http://en.wikipedia.org/wiki/Eigenvalue_algorithm
					//Oliver K. Smith: Eigenvalues of a symmetric 3 × 3 matrix. Commun. ACM 4(4): 168 (1961)
					float a0, b0, c0, d0, e0, f0;
					a0 = -d_gxx[i]; b0 = -d_gxy[i]; c0 = -d_gxz[i];
					d0 = -d_gyy[i]; e0 = -d_gyz[i]; f0 = -d_gzz[i];

					float m = (a0+d0+f0)/3;
					float q = computeDeterminant
						(a0-m, b0, c0, b0, d0-m, e0, c0, e0, f0-m)/2;
					float p = (a0-m)*(a0-m) + b0*b0 + c0*c0 + b0*b0 + (d0-m)*(d0-m) +
						e0*e0 + c0*c0 + e0*e0 + (f0-m)*(f0-m);
					p = p / 6;
					float phi = 1.f/3.f*atan(sqrt(p*p*p-q*q)/q);
					if(phi<0)
						phi=phi+3.14159f/3;

					float eig1 = m + 2*sqrt(p)*cos(phi);
					float eig2 = m - sqrt(p)*(cos(phi) + sqrt(3.0f)*sin(phi));
					float eig3 = m - sqrt(p)*(cos(phi) - sqrt(3.0f)*sin(phi));

					if( (eig1 > eig2) & (eig1 > eig3))
						d_output[i] = eig1*sigma*sigma;
					if( (eig2 > eig1) & (eig2 > eig3))
						d_output[i] = eig2*sigma*sigma;
					if( (eig3 > eig2) & (eig3 > eig1))
						d_output[i] = eig3*sigma*sigma;


					// // Now it comes to compute the eigenvector
					float l = d_output[i]/(sigma*sigma);
					a0 = a0 - l;
					d0 = d0 - l;
					f0 = f0 - l;
					float xv = b0*e0 - c0*d0;
					float yv = e0*a0 - c0*b0;
					float zv = d0*a0 - b0*b0;
					float radius = sqrt(xv*xv+yv*yv+zv*zv);
					float thetav = atan2(yv, xv);
					float phiv = 0;
					if(radius > 1e-6f)
						phiv = acos( zv/radius);

					d_output_theta[i] = thetav;
					d_output_phi[i] = phiv;

				}


			void hessian_orientation
				(
				 cuv::tensor<float,dev_memory_space> &d_Output,
				 cuv::tensor<float,dev_memory_space> &d_Output_theta,
				 cuv::tensor<float,dev_memory_space> &d_Output_phi,
				 const cuv::tensor<float,dev_memory_space> &d_gxx,
				 const cuv::tensor<float,dev_memory_space> &d_gxy,
				 const cuv::tensor<float,dev_memory_space> &d_gxz,
				 const cuv::tensor<float,dev_memory_space> &d_gyy,
				 const cuv::tensor<float,dev_memory_space> &d_gyz,
				 const cuv::tensor<float,dev_memory_space> &d_gzz,
				 float sigma
				)
				{
					int imageW = d_gxx.shape()[2];
					int imageH = d_gxx.shape()[1];
					int imageD = d_gxx.shape()[0];
					dim3 grid (imageD*imageW/ROWS_BLOCKDIM_X,imageH/ROWS_BLOCKDIM_Y);
					dim3 block(ROWS_BLOCKDIM_X,ROWS_BLOCKDIM_Y);
					hessianKernelO<<<grid, block>>>( d_Output.ptr(), d_Output_theta.ptr(), d_Output_phi.ptr(),
							d_gxx.ptr(), d_gxy.ptr(), d_gxz.ptr(),
							d_gyy.ptr(), d_gyz.ptr(), d_gzz.ptr(),
							sigma, imageW, imageH, imageD );
					cuvSafeCall(cudaThreadSynchronize());
				}

		}
	}
}
