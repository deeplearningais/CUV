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

#include <cuv/basics/tensor.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/libs/separable_conv/separable_convolution.hpp>
#include <cuv/libs/nlmeans/conv3d.hpp>
#include "hog.hpp"

template<class V, class I>
__global__
void select_arg_kernel(V* dst, const V* src, const I* arg, unsigned int w){
	const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id>=w) return;
	dst[id] = src[arg[id]*w+id];
}

template<class V>
__global__
void atan2_abs_kernel(V* dst, const V* gy, const V* gx, unsigned int w){
	const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id>=w) return;
	V tmp = atan2(gy[id],gx[id]);
	if(tmp<0) tmp += (float) M_PI;
	dst[id] = tmp;
}


template<class V>
__global__
void orientation_binning_kernel(V*dst, const V* norms, const V* angles, const unsigned int steps, const unsigned int w, const unsigned int h){
	const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	V lowerbinval = 0;
	V upperbinval = 0;
	V ang = angles[id];
	V mag = norms [id];
	const V angdiff = (float)M_PI / (float)steps;
	for(unsigned int s = 0; s<steps; s++){
		float a1 = (float)M_PI / steps * s;
		float a2 = (float)M_PI / steps * (s+1);
		if(ang>=a1 && ang<a2){
			lowerbinval += mag * (a2-ang)/angdiff;
			upperbinval  = mag * (ang-a1)/angdiff;
		}else{
			upperbinval = 0;
		}
		// for linear interpolation, we write exactly twice in each bin. 
		// after the second time we write the lower value.
		// this works for all except the first iterations of s (which
		// is therefore written twice to global memory, see after loop)
		dst[s*w*h + id] = lowerbinval;
		lowerbinval     = upperbinval;
	}

	// this slightly suboptimal operation saves lots of juggling with shared memory.
	dst[0*w*h + id] += lowerbinval;
}



namespace cuv{ namespace libs{ namespace hog{

	namespace detail{
		inline unsigned int __host__ __device__ divup(unsigned int a, unsigned int b)
		{
			if (a % b)  /* does a divide b leaving a remainder? */
				return a / b + 1; /* add in additional block */
			else
				return a / b; /* divides cleanly */
		}


		/** after determining the argmax using reduce, we need to extract the
		 *  corresponding values from two matrices. This kernel does the job.
		 *
		 * @param dst     h x w matrix where result is written to
		 * @param src c x h x w matrix where we read from
		 * @param arg       h*w matrix of indices, elements should be in [0,c[
		 *
		 */
		template<class V, class I>
		void select_arg(cuv::tensor<V,dev_memory_space>& dst, const cuv::tensor<V,dev_memory_space>&src, const cuv::tensor<I,dev_memory_space>& arg){
			cuvAssert(dst.ndim()==2);
			cuvAssert(src.ndim()==3);
			cuvAssert(arg.ndim()==1);
			const unsigned int h = dst.shape()[0];
			const unsigned int w = dst.shape()[1];
			cuvAssert(src.shape()[1]==h);
			cuvAssert(src.shape()[2]==w);
			cuvAssert(arg.shape()[0]==h*w);
			/*cuvAssert(minimum(arg)>=0);*/
			/*cuvAssert(maximum(arg)< src.shape()[0]);*/
			dim3 blocks(divup(arg.shape()[0], 256));
			dim3 threads(256);
			select_arg_kernel<<<blocks,threads>>>(dst.ptr(),src.ptr(),arg.ptr(),arg.shape()[0]);
			cuvSafeCall(cudaThreadSynchronize());
		}

		/** determine angle of gradient disregarding polarisation
		  */
		template<class V>
		void atan2_abs(cuv::tensor<V,dev_memory_space>& dst, const cuv::tensor<V,dev_memory_space>& gy, const cuv::tensor<V,dev_memory_space>& gx){
			cuvAssert(equal_shape(gx,gy));
			cuvAssert(equal_shape(gx,dst));
			dim3 blocks(divup(dst.size(),256));
			dim3 threads(256);
			atan2_abs_kernel<<<blocks,threads>>>(dst.ptr(),gy.ptr(),gx.ptr(),dst.size());
			cuvSafeCall(cudaThreadSynchronize());
		}

		/** bin orientations with bilinear interpolation
		  * @param dst    steps x h x w matrix of resulting gradient maps
		  * @param norms          h x w matrix of gradient magnitudes
		  * @param angles         h x w matrix of angles to be binned
		  */
		template<class V>
		void orientation_binning(cuv::tensor<V,dev_memory_space>& dst, const cuv::tensor<V,dev_memory_space>& norms, const cuv::tensor<V,dev_memory_space>& angles){
			cuvAssert(dst.ndim()==3);
			cuvAssert(norms.ndim()==2);
			cuvAssert(equal_shape(norms,angles));
			cuvAssert(dst.shape()[1]==norms.shape()[0]);
			cuvAssert(dst.shape()[2]==norms.shape()[1]);
			const unsigned int steps = dst.shape()[0];
			const unsigned int h     = dst.shape()[1];
			const unsigned int w     = dst.shape()[2];
			dim3 blocks(divup(dst.shape()[1]*dst.shape()[2],256));
			dim3 threads(256);
			orientation_binning_kernel<<<blocks,threads>>>(dst.ptr(),norms.ptr(),angles.ptr(),steps,w,h);
			cuvSafeCall(cudaThreadSynchronize());
		}

		template<class V>
			void hog(cuv::tensor<V, dev_memory_space>& bins, const cuv::tensor<V,dev_memory_space>& src, unsigned int spatialpool){
				typedef cuv::tensor<V,dev_memory_space> tens_t;

				unsigned int chann  = src.shape()[0];
				unsigned int height = src.shape()[1];
				unsigned int width  = src.shape()[2];
				unsigned int steps  = bins.shape()[0];


				tens_t magnitude(extents[height][width]);
				tens_t angle    (extents[height][width]);
				{       tens_t  gradx(src.shape()), 
						grady(src.shape()), 
						allmagnitudes(src.shape()),
						allangles(src.shape());

					// determine the centered derivatives in x and y direction
					cuv::tensor<float,host_memory_space> diff(3);
					diff[0] = -0.5f;
					diff[1] =  0.f;
					diff[2] =  0.5f;
					cuv::libs::nlmeans::setConvolutionKernel_horizontal(diff);
					cuv::libs::nlmeans::setConvolutionKernel_vertical(diff);

					cuv::libs::nlmeans::convolutionRows(grady,src,1);
					cuv::libs::nlmeans::convolutionColumns(gradx,src,1);

					// calculate the gradient norms and directions
					cuv::apply_binary_functor(allmagnitudes, gradx,grady, BF_NORM);
					atan2_abs(allangles, gradx,grady);

					// determine channel with maximal magnitude
					allmagnitudes.reshape(chann,height*width);
					cuv::tensor<unsigned int,dev_memory_space> argmax(extents[height*width]);
					cuv::reduce_to_row(argmax,allmagnitudes,RF_ARGMAX);
					allmagnitudes.reshape(extents[chann][height][width]);

					// in magnitudes/angles, put maximal values
					select_arg(magnitude,allmagnitudes,argmax);
					select_arg(angle    ,allangles    ,argmax);
				}

				// discretize using bilinear interpolation
				orientation_binning(bins,magnitude,angle);

				// spatial pooling
				{
					cuv::tensor<float,host_memory_space> kernel(2*spatialpool+1);
					float sigma = spatialpool/2.f;
					for(int i = 0; i < 2*spatialpool+1; i++){
						float dist = (float)(i - (int)spatialpool);
						kernel[i]  = expf(- dist * dist / (2*sigma*sigma));
					}
					kernel /= cuv::sum(kernel);
					cuv::libs::nlmeans::setConvolutionKernel_horizontal(kernel);
					cuv::libs::nlmeans::setConvolutionKernel_vertical(kernel);

					tens_t intermed(bins.shape());
					cuv::libs::nlmeans::convolutionRows(intermed,bins,spatialpool);
					cuv::libs::nlmeans::convolutionColumns(bins,intermed,spatialpool);
				}

				// normalization
				bins.reshape(extents[steps][width*height]);
				tens_t norms(width*height);
				reduce_to_row(norms,bins,RF_ADD_SQUARED);
				norms += 0.0001f;
				apply_scalar_functor(norms,SF_SQRT);
				matrix_divide_row(bins,norms);

				// clip
				apply_scalar_functor(bins,SF_MIN, 0.2f);

				// renormalize
				reduce_to_row(norms,bins,RF_ADD_SQUARED);
				norms += 0.0001f;
				apply_scalar_functor(norms,SF_SQRT);
				matrix_divide_row(bins,norms);

				bins.reshape(extents[steps][width][height]);
			}
		template<class V>
			void hog(cuv::tensor<V, host_memory_space>& dst, const cuv::tensor<V,host_memory_space>& src, unsigned int spatialpool){
				throw std::runtime_error("not implemented");
			}
	}
	template<class V, class M>
	void hog(cuv::tensor<V, M>& dst, const cuv::tensor<V,M>& src, unsigned int spatialpool){
		cuvAssert(src.ndim()==3);
		cuvAssert(dst.ndim()==3);

		cuvAssert(src.shape()[0]==3);
		cuvAssert(src.shape()[1]==dst.shape()[1]);
		cuvAssert(src.shape()[2]==dst.shape()[2]);

		detail::hog(dst,src, spatialpool);
	}

#define TENS(V,M) \
        cuv::tensor<V,M>
#define INSTANTIATE(V,M) \
	template void hog(TENS(V, M)&, const TENS(V, M)&, unsigned int);

INSTANTIATE(float,dev_memory_space)
	
			
} } }
