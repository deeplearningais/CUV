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





/** 
 * @file convolution_ops.cu
 * @brief Operations used for convolution and max-pooling
 * @ingroup convolution
 * @date 2010-03-21
 */

#include <cuv/basics/tensor.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/random/random.hpp>
#include <3rd_party/cudaconv2/include/cudaconv2/conv_util.cuh>
#include <3rd_party/cudaconv2/include/cudaconv2/cudaconv2.cuh>
#include <3rd_party/cudaconv2/include/nvmatrix/nvmatrix.cuh>
/*#include <3rd_party/cudaconv2/include/convCPU.h>*/
#include <cuv/convolution_ops/convolution_ops.hpp>

#define NVView3D(X)  \
        (const_cast<float*>(X.ptr()), X.shape(0)*X.shape(1), X.shape(2), X.shape(2),false)

namespace cuv{ namespace alex_conv{

template<class V,class M, class T>
    void reorder_for_conv(tensor<V,M,T>& dst, const tensor<V,M,T>& src){
        cuvAssert(src.ndim()==3);
        cuvAssert(dst.ndim()==3);
        std::vector<unsigned int> s = src.shape();
        /*tensor<V,M,T> src_view(indices[index_range()][index_range()][index_range()], src);*/
        tensor<V,M,T>& src_view  = const_cast<tensor<V,M,T>&>(src);
        src_view.reshape(extents[s[0]][s[1]*s[2]]);
        dst.reshape(extents[s[1]*s[2]][s[0]]);
        cuv::transpose(dst,src_view);
        src_view.reshape(s);
        dst.reshape(extents[s[1]][s[2]][s[0]]);
    }
template<class V,class M, class T>
    void reorder_from_conv(tensor<V,M,T>& dst, const tensor<V,M,T>& src){
        cuvAssert(src.ndim()==3);
        cuvAssert(dst.ndim()==3);
        tensor<V,M,T> src_view(indices[index_range()][index_range()][index_range()], src);
        src_view.reshape(extents[src.shape(0)*src.shape(1)][src.shape(2)]);
        dst.reshape(extents[dst.shape(0)][dst.shape(1)*dst.shape(2)]);
        cuv::transpose(dst,src_view);
        dst.reshape(extents[src.shape(2)][src.shape(0)][src.shape(1)]);
    }

template<>
    void 
    convolve2d(tensor<float,dev_memory_space>& dst, 
            const tensor<float,dev_memory_space>& img, 
            const tensor<float,dev_memory_space>& filter,
            unsigned int paddingStart, 
            unsigned int moduleStride,
            unsigned int nGroups){
        // check compatibility before converting to NVMatrix format
        /*cuvAssert(dst.ndim()==3);*/
        cuvAssert(img.ndim()==3);
        unsigned int nImgChan = img.shape(0);
        unsigned int nImgPix  = img.shape(1);
        unsigned int nImg     = img.shape(2);

        cuvAssert(filter.ndim()==3);
        unsigned int nFiltChan = filter.shape(0);
        unsigned int nFiltPix  = filter.shape(1);
        unsigned int nFilt     = filter.shape(2);

        cuvAssert(dst.shape(0)==nFilt);
        unsigned int nModules = dst.shape(1);
        unsigned int nModulesX = sqrt(nModules);
        cuvAssert(nModules == nModulesX * nModulesX);
        cuvAssert(dst.shape(2)==nImg);

        // make NVMatrices with this data
        NVMatrix nv_dst    NVView3D(dst);
        NVMatrix nv_img    NVView3D(img);
        NVMatrix nv_filter NVView3D(filter);

        if(nFilt<16){
            // we can use this for output maps, which still must be divisible by four(!)
            // this is still fully connected, however we must resort to "sparse" conv
            // since the non-sparse conv only allows 
            int* colorIndices = new int[nGroups*nFiltChan]; 
            for(unsigned int i=0;i<nGroups*nFiltChan;i++) colorIndices[i]=i;
            convFilterActsSparse(nv_img, nv_filter, nv_dst, colorIndices, nModulesX, paddingStart, moduleStride, nImgChan, nFiltChan, nGroups);
        }{
            convFilterActs(nv_img, nv_filter, nv_dst, nModulesX, paddingStart, moduleStride, nImgChan, nGroups);
        }
    }
template<>
	void d_conv2d_dimg(tensor<float,dev_memory_space,row_major>& dst,
			  const tensor<float,dev_memory_space,row_major>&   delta,
			  const tensor<float,dev_memory_space,row_major>&   filter,
              unsigned int paddingStart, unsigned int moduleStride, unsigned int nGroups){


        cuvAssert(delta.ndim()==3);
        unsigned int nFilt    = delta.shape(0);
        unsigned int nModules = delta.shape(1); 
        unsigned int nImg     = delta.shape(2);

        cuvAssert(filter.ndim()==3);
        unsigned int nFiltChan = filter.shape(0);
        unsigned int nFiltPix  = filter.shape(1);
        /*unsigned int nFilt     = filter.shape(2);*/
        cuvAssert(filter.shape(2) == nFilt);

        cuvAssert(dst.ndim()==3);
        unsigned int nImgChan  = dst.shape(0);
        unsigned int nImgPix   = dst.shape(1);
        cuvAssert(dst.shape(2) == nImg);

        unsigned int imgSize = sqrt(nImgPix);
        cuvAssert(nImgPix == imgSize*imgSize);

        /*void convImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,*/
        /*    int imgSize, int paddingStart, int moduleStride, int numImgColors, int numGroups);*/

        NVMatrix nv_dst    NVView3D(dst);
        NVMatrix nv_delta  NVView3D(delta);
        NVMatrix nv_filter NVView3D(filter);

        convImgActs(nv_delta, nv_filter, nv_dst,
                imgSize, paddingStart, moduleStride, nImgChan, nGroups);
    }
template<>
	void d_conv2d_dfilt(tensor<float,dev_memory_space,row_major>& dst_,
			  const tensor<float,dev_memory_space,row_major>&   delta,
			  const tensor<float,dev_memory_space,row_major>&   input,
              unsigned int paddingStart,
            unsigned int moduleStride, unsigned int nGroups, unsigned int partialSum){

        cuvAssert(dst_.ndim()==3);
        unsigned int nFiltChan = dst_.shape(0);
        unsigned int nFiltPix  = dst_.shape(1);
        unsigned int nFilt     = dst_.shape(2);



        unsigned int filtSize = sqrt(nFiltPix);
        cuvAssert ( nFiltPix == filtSize*filtSize );


        cuvAssert(delta.ndim()==3);
        cuvAssert(delta.shape(0) == nFilt);
        unsigned int nModules  = delta.shape(1);
        unsigned int nImg      = delta.shape(2);

        unsigned int nModulesX = sqrt(nModules);
        cuvAssert(nModules == nModulesX * nModulesX);

        cuv::tensor<float,dev_memory_space> dst(extents[nModules/partialSum][nFiltChan][nFiltPix][nFilt]);

        cuvAssert(input.ndim()==3);
        unsigned int nImgChan = input.shape(0);
        unsigned int nImgPix  = input.shape(1);
        cuvAssert(input.shape(2) == nImg);

        unsigned int imgSize = sqrt(nImgPix);
        cuvAssert(nImgPix == imgSize*imgSize);


        /*void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,*/
        /*                    int numModulesX, int filterSize, int paddingStart,*/
        /*                    int moduleStride, int numImgColors, int numGroups, int partialSum);*/
        NVMatrix nv_dst   NVView3D(dst);
        NVMatrix nv_delta NVView3D(delta);
        NVMatrix nv_input NVView3D(input);
        convWeightActs(nv_input, nv_delta, nv_dst,
                nModulesX, filtSize, paddingStart,
                moduleStride, nImgChan, nGroups, partialSum);

        dst.reshape(extents[nModules/partialSum][nFiltChan*nFiltPix*nFilt]);
        dst_.reshape(extents[nFiltChan*nFiltPix*nFilt]);
        cuv::reduce_to_row(dst_,dst);
        dst_.reshape(extents[nFiltChan][nFiltPix][nFilt]);
    }


template<>
    void local_pool(tensor<float,dev_memory_space>& target,
            const tensor<float,dev_memory_space>& images,
            int subsX, int startX, int strideX, int outputsX, pool_type pooler){

        cuvAssert(images.ndim()==3);
        unsigned int nFilt   = images.shape(0);
        unsigned int nImgPix = images.shape(1);
        unsigned int nImg    = images.shape(2);

        cuvAssert(target.ndim()==3);
        cuvAssert(target.shape(0) == nFilt);
        unsigned int outputs = target.shape(1);
        cuvAssert(target.shape(2) == nImg);

        unsigned int imgSize = sqrt(nImgPix);
        cuvAssert(imgSize*imgSize == nImgPix);

        unsigned int outSize = sqrt(outputs);
        cuvAssert(outSize*outSize == outputs);

        unsigned int poolSize = imgSize / outSize;
        cuvAssert(poolSize*outSize == imgSize);

        NVMatrix nv_target NVView3D(target);
        NVMatrix nv_images NVView3D(images);
        

        switch(pooler){
            case PT_MAX:
                convLocalPool(nv_images, nv_target, nFilt,
                        subsX, startX, strideX, outputsX, MaxPooler());
                break;
            case PT_AVG:
                convLocalPool(nv_images, nv_target, nFilt,
                        subsX, startX, strideX, outputsX, AvgPooler(poolSize));
                break;
        }
    }
template<>
    void local_max_pool_grad(tensor<float,dev_memory_space>& target, const tensor<float,dev_memory_space>& images, const tensor<float,dev_memory_space>& maxGrads,
            const tensor<float,dev_memory_space>& maxActs, int subsX, int startX, int strideX){


        cuvAssert(target.ndim()==3);
        unsigned int nImgChan  = target.shape(0);
        unsigned int nImgPix   = target.shape(1);
        unsigned int nImg      = target.shape(2);

        cuvAssert(images.ndim()==3);
        cuvAssert(nImgChan == images.shape(0));
        unsigned int nOutPix = images.shape(1);
        cuvAssert(nImg     == images.shape(2));

        unsigned int outputsX = sqrt(nOutPix);
        cuvAssert(outputsX*outputsX==nOutPix);

        NVMatrix nv_target NVView3D(target);
        NVMatrix nv_images NVView3D(images);
        NVMatrix nv_maxGrads NVView3D(maxGrads);
        NVMatrix nv_maxActs NVView3D(maxActs);
        
/*void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,*/
/*                      int subsX, int startX, int strideX, int outputsX);*/
        convLocalMaxUndo(nv_images,nv_maxGrads, nv_maxActs, nv_target, subsX,startX,strideX,outputsX);
    }

template<>
    void local_avg_pool_grad(tensor<float,dev_memory_space>& target, const tensor<float,dev_memory_space>& avgGrads,
            int subsX, int startX, int strideX){


        cuvAssert(target.ndim()==3);
        unsigned int nImgChan  = target.shape(0);
        unsigned int nImgPix   = target.shape(1);
        unsigned int nImg      = target.shape(2);

        cuvAssert(avgGrads.ndim()==3);
        cuvAssert(nImgChan == avgGrads.shape(0));
        unsigned int nOutPix = avgGrads.shape(1);
        cuvAssert(nImg == avgGrads.shape(2));

        unsigned int outputsX = sqrt(nOutPix);
        cuvAssert(outputsX*outputsX==nOutPix);

        unsigned int imgX = sqrt(nImgPix);
        cuvAssert(imgX*imgX == nImgPix);

        NVMatrix nv_target NVView3D(target);
        NVMatrix nv_avgGrads NVView3D(avgGrads);
        
        convLocalAvgUndo(nv_avgGrads, nv_target, subsX,startX,strideX,outputsX,imgX);
    }

// instantiate
#define  TENS(V,M,T)       tensor<V,M,T>
#define CTENS(V,M,T) const TENS(V,M,T)
#define INST(V,M,T) \
template void reorder_for_conv<V,M,T>(TENS(V,M,T)&, CTENS(V,M,T)&); \
template void reorder_from_conv<V,M,T>(TENS(V,M,T)&, CTENS(V,M,T)&);
INST(float,host_memory_space,row_major);
INST(float,dev_memory_space,row_major);
}}

