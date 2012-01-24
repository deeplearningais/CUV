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
#include <iostream>
#include <stdlib.h>
#include <dlfcn.h>
#include <vector>
#include "integral_image.hpp"

#define NUM_BANKS 16  
#define LOG_NUM_BANKS 4  
#define CONFLICT_FREE_OFFSET(x) ((x) >> NUM_BANKS + (x) >> (2 * LOG_NUM_BANKS))
#define PITCH(PTR,PITCH,Y,X) ((typeof(PTR))((char*)(PTR) + (PITCH)*(Y)) + (X))

namespace cuv
{
namespace integral_img{
	template<int BLOCK_SIZE,class DstT, class SrcT, class I>
		__global__ void scan_kernel(DstT *output, const SrcT *input, I width, I dpitch, I spitch) {
			__shared__ DstT sum1, sum2;

			I blockCol = blockIdx.x;
			I tdx      = threadIdx.x;
			if(tdx==0)
				sum1 = sum2 = 0;

			const SrcT* src = PITCH(input, spitch,blockCol,0);
			      DstT* dst = PITCH(output,dpitch,blockCol,0);

			      for(I i = 0; i< width; i+=BLOCK_SIZE) {
				      __syncthreads();
				      __shared__ DstT temp[BLOCK_SIZE];
				      I offset =1;

				      temp[2*tdx]   = (i+2*tdx  )<width ? src[i+2*tdx]   : 0;
				      temp[2*tdx+1] = (i+2*tdx+1)<width ? src[i+2*tdx+1] : 0;

				      for(I outerd = BLOCK_SIZE/2; outerd > 0; outerd /= 2) {
					      __syncthreads();
					      if(tdx < outerd) {
						      I ai      = offset*(2*tdx+1)-1;
						      I bi      = offset*(2*tdx+2)-1;
						      temp[bi] += temp[ai];
					      }
					      offset   *= 2;

				      }
				      if(tdx == 0) {
					      sum2               = temp[BLOCK_SIZE-1];
					      temp[BLOCK_SIZE-1] = 0;
				      }
				      for(I innerd = 1; innerd < BLOCK_SIZE; innerd *= 2) {
					      offset >>= 1;
					      __syncthreads();
					      if(tdx < innerd) {
						      I ai      = offset*(2*tdx+1)-1;
						      I bi      = offset*(2*tdx+2)-1;
						      DstT t    = temp[ai];
						      temp[ai]  = temp[bi];
						      temp[bi] += t;
					      }
				      }
				      __syncthreads();

				      if(i+2*tdx  <width) dst[i+2*tdx]   = temp[2*tdx]  +sum1;
				      if(i+2*tdx+1<width) dst[i+2*tdx+1] = temp[2*tdx+1]+sum1;
				      if(tdx == 0) 
					      sum1 += sum2;
			      }
		}

	template<class V,class W, class L>
		void scan(cuv::tensor<V, dev_memory_space, L>& dst, const cuv::tensor<W, dev_memory_space, L>& src) {
			scan_kernel<256,V><<<src.shape()[0], 128>>>(dst.ptr(), src.ptr(), src.shape(1), (unsigned int)dst.stride(0), (unsigned int)src.stride(0));
			cuvSafeCall(cudaThreadSynchronize());
		}

	template<class V,class W, class L>
		void scan(cuv::tensor<V, host_memory_space, L>& dst, const cuv::tensor<W, host_memory_space, L>& src)
		{
			const W* src_ptr = src.ptr();
			V* dst_ptr = dst.ptr();
			for(int i = 0; i<src.shape()[0]; i++) {
				*dst_ptr = 0;
				dst_ptr++;
				for(int j =0; j< src.shape()[1]-1; j++) {
					*dst_ptr = *(dst_ptr-1) + *(src_ptr);
					dst_ptr++;
					src_ptr++;
				}
				src_ptr++;
			}
		}

	template<class V,class W, class T, class M>
		void integral_image(cuv::tensor<V, T, M>& dst, const cuv::tensor<W, T, M>& src)
		{
			cuvAssert(src.ndim()==2);
			cuvAssert(src.shape()[0]==dst.shape()[1]);
			cuvAssert(src.shape()[1]==dst.shape()[0]);
			tensor<V,T,M> temp (src.shape(),pitched_memory_tag());
			tensor<V,T,M> temp1(dst.shape(),pitched_memory_tag());

			scan(temp, src);
			transpose(temp1, temp);
			scan(dst, temp1);
		}
#define TENS(V,M,L) \
        cuv::tensor<V,M,L>
#define INSTANTIATE_INTIMG(V,W,M,L) \
	template void integral_image(TENS(V, M, L)&, const TENS(W, M, L)&);\
	template void scan(TENS(V          , M, L)&, const TENS(W, M, L)&);

	INSTANTIATE_INTIMG(float, float        , host_memory_space, row_major);
	INSTANTIATE_INTIMG(float, unsigned char, host_memory_space, row_major);
	INSTANTIATE_INTIMG(float, float        , dev_memory_space , row_major);
	INSTANTIATE_INTIMG(float, unsigned char, dev_memory_space , row_major);


} // namespace integral image
} // namespace cuv



