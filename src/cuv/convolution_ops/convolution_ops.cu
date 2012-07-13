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
#define NVView4D(X)  \
        (const_cast<float*>(X.ptr()), X.shape(0)*X.shape(1)*X.shape(2), X.shape(3), X.shape(3),false)

namespace cuv{ namespace alex_conv{

template<class V,class M, class T>
    void reorder_for_conv(tensor<V,M,T>& dst, const tensor<V,M,T>& src){
        cuvAssert(src.ndim()==4);
        cuvAssert(dst.ndim()==4);
        std::vector<unsigned int> s = src.shape();
        /*tensor<V,M,T> src_view(indices[index_range()][index_range()][index_range()], src);*/
        tensor<V,M,T>& src_view  = const_cast<tensor<V,M,T>&>(src);
        src_view.reshape(extents[s[0]][s[1]*s[2]*s[3]]);
        dst.reshape(extents[s[1]*s[2]*s[3]][s[0]]);
        cuv::transpose(dst,src_view);
        src_view.reshape(s);
        dst.reshape(extents[s[1]][s[2]][s[3]][s[0]]);
    }
template<class V,class M, class T>
    void reorder_from_conv(tensor<V,M,T>& dst, const tensor<V,M,T>& src){
        cuvAssert(src.ndim()==4);
        cuvAssert(dst.ndim()==4);
        tensor_view<V,M,T> src_view(indices[index_range()][index_range()][index_range()][index_range()], src);
        src_view.reshape(extents[src.shape(0)*src.shape(1)*src.shape(2)][src.shape(3)]);
        dst.reshape(extents[src.shape(3)][src.shape(0)*src.shape(1)*src.shape(2)]);
        cuv::transpose(dst,src_view);
        dst.reshape(extents[src.shape(3)][src.shape(0)][src.shape(1)][src.shape(2)]);
    }

/*
 * hidActs:     (numFilters, numModules, numImages)
 * filters:     (numFilterColors, filterPixels, numFilters)               if conv
 *              (numModules, numFilterColors, filterPixels, numFilters)   otherwise
 * targets:     (numImageColors, imgPixels, numImages)
 */
void cpuImgActs(const float* hidActs, const float* filters, float* targets,
               int numModulesX,  int numImages,  int numFilters,
               int filterSize,  int imgSize,  int moduleStart,
               int moduleStride, int numImgColors, int numGroups, bool conv) {
    int filterPixles = filterSize * filterSize;
    int imgPixels = imgSize * imgSize;
    int numModules = numModulesX * numModulesX;
    int numFiltersPerGroup = numFilters / numGroups;
    int numFilterColors = numImgColors / numGroups;
    for (int py = 0; py < imgSize; py++) {
        for (int px = 0; px < imgSize; px++) {
            for (int my = 0; my < numModulesX; my++) {
                int moduleTop = moduleStart + my * moduleStride;
                int moduleBottom = moduleTop + filterSize;
                for (int mx = 0; mx < numModulesX; mx++) {
                    int m = my * numModulesX + mx;
                    int moduleLeft = moduleStart + mx * moduleStride;
                    int moduleRight = moduleLeft + filterSize;
                    int pixInModuleX = px - moduleLeft;
                    int pixInModuleY = py - moduleTop;
                    int pixInModule = pixInModuleY * filterSize + pixInModuleX;
                    if (py >= moduleTop && py < moduleBottom && px >= moduleLeft && px < moduleRight) {
                        for (int f = 0; f < numFilters; f++) {
                            int g = f / numFiltersPerGroup; // filter's group idx
                            for (int i = 0; i < numImages; i++) {
                                for (int c = 0; c < numFilterColors; c++) {
                                    float w = filters[(conv ? 0 : m * numFilterColors * filterPixles * numFilters) 
                                                      + c * numFilters * filterPixles + pixInModule * numFilters + f];
                                    float h = hidActs[m * numImages + f * numModules * numImages + i];
                                    targets[(c + g * numFilterColors) * imgPixels * numImages + i] += w * h;
                                }
                            }

                        }
                    }
                }
            }
            targets += numImages;
        }
    }
}

/*
 * images:      (numImgColors, imgPixels, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters)             if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModules, numImages)
 */
void cpuFilterActs(const float* images, const float* filters, float* targets,
                       int numImages, int numFilters,
                       int imgSize, int filterSize, int paddingStart,
                       int moduleStride, int numModulesX,
                       int numImgColors, int numGroups, bool conv, float scaleTargets, float scaleOutput) {
    int filterPixels = filterSize * filterSize;
    int numFilterColors = numImgColors / numGroups;
    int numModules = numModulesX * numModulesX;
    int imgPixels = imgSize * imgSize;
    int groupColorStride = numGroups == 1 ? 0 : (numImgColors - numFilterColors) / (numGroups - 1);
    int filtersPerGroup = numFilters / numGroups;
    for (int my = 0; my < numModulesX; my++) {
        int mStartY = paddingStart + my * moduleStride;
        for (int mx = 0; mx < numModulesX; mx++) {
            int mStartX = paddingStart + mx * moduleStride;
            int m = (my * numModulesX + mx);
            for (int f = 0; f < numFilters; f++) {
                int g = f / filtersPerGroup; // filter group
                for (int i = 0; i < numImages; i++) {
                    float prod = 0;
                    for (int c = 0; c < numFilterColors; c++) {
                        for (int y = 0; y < filterSize; y++) {
                            for (int x = 0; x < filterSize; x++) {
                                float imgVal = mStartY + y >= 0 && mStartY + y < imgSize && mStartX + x >= 0 && mStartX + x < imgSize
                                            ? images[(c + g * groupColorStride) * imgPixels * numImages + i + ((mStartY+y) * imgSize + mStartX+x) * numImages]
                                            : 0;
                                float fVal = filters[c * filterPixels * numFilters + f + (y * filterSize + x) * numFilters
                                                     + (conv ? 0 : m * numFilters * filterPixels * numFilterColors)];
                                prod += fVal * imgVal;
                            }
                        }
                    }

                    targets[f * numModules * numImages + m * numImages + i] = scaleTargets*targets[f * numModules * numImages + m * numImages + i] + scaleOutput* prod;
                }
            }
        }
    }
}

template<class V, class M, class T>
    void 
    convolve2d(tensor<V,M, T>& dst, 
            const tensor<V,M, T>& img, 
            const tensor<V,M, T>& filter,
            int paddingStart, 
            unsigned int moduleStride,
            unsigned int nGroups,
            float factNew,
            float factOld){
        // check compatibility before converting to NVMatrix format
        /*cuvAssert(dst.ndim()==3);*/
        cuvAssert(img.ndim()==4);
        unsigned int nImgChan  = img.shape(0);
        unsigned int nImgPixY  = img.shape(1);
        unsigned int nImgPixX  = img.shape(2);
        unsigned int nImg      = img.shape(3);

        cuvAssert(filter.ndim()==3);
        unsigned int nFiltChan = filter.shape(0);
        unsigned int nFiltPix  = filter.shape(1);
        unsigned int nFilt     = filter.shape(2);

        cuvAssert(dst.shape(0)==nFilt);
        unsigned int nModulesY = dst.shape(1);
        unsigned int nModulesX = dst.shape(2);
        cuvAssert(dst.shape(3)==nImg);

        // make NVMatrices with this data
        NVMatrix nv_dst    NVView4D(dst);
        NVMatrix nv_img    NVView4D(img);
        NVMatrix nv_filter NVView3D(filter);

        if(nFilt<16){
            // we can use this for output maps, which still must be divisible by four(!)
            // this is still fully connected, however we must resort to "sparse" conv
            // since the non-sparse conv only allows 
            int* colorIndices = new int[nGroups*nFiltChan]; 
            for(unsigned int i=0;i<nGroups*nFiltChan;i++) colorIndices[i]=i;
            convFilterActsSparse(nv_img, nv_filter, nv_dst, colorIndices, nImgPixY, nModulesY, nModulesX, paddingStart, moduleStride, nImgChan, nFiltChan, nGroups,factOld,factNew);
        }{
            if(IsSame<M,dev_memory_space>::Result::value){
                convFilterActs(nv_img, nv_filter, nv_dst, nImgPixY, nModulesY, nModulesX, paddingStart, moduleStride, nImgChan, nGroups, factOld,factNew);
            }else{
                unsigned int filtX  = sqrt(nFiltPix);
                cuvAssert(filtX*filtX == nFiltPix);

                cpuFilterActs(img.ptr(), filter.ptr(), dst.ptr(), 
                        nImg, nFilt, 
                        nImgPixX, filtX, paddingStart,
                        moduleStride, nModulesX, 
                        nImgChan, nGroups, true, factOld,factNew);
            }
        }
    }
template<class V, class M, class L>
	void d_conv2d_dimg(tensor<V,M,L>& dst,
			  const tensor<V,M,L>&   delta,
			  const tensor<V,M,L>&   filter,
              int paddingStart, unsigned int moduleStride, unsigned int nGroups, float factNew,float factOld){


        cuvAssert(delta.ndim()==4);
        unsigned int nFilt     = delta.shape(0);
        unsigned int nModulesY = delta.shape(1); 
        unsigned int nModulesX = delta.shape(2); 
        unsigned int nImg      = delta.shape(3);

        cuvAssert(filter.ndim()==3);
        unsigned int nFiltChan = filter.shape(0);
        unsigned int nFiltPix  = filter.shape(1);
        /*unsigned int nFilt     = filter.shape(2);*/
        cuvAssert(filter.shape(2) == nFilt);

        cuvAssert(dst.ndim()==4);
        unsigned int nImgChan  = dst.shape(0);
        unsigned int nImgPixY  = dst.shape(1);
        unsigned int nImgPixX  = dst.shape(2);
        cuvAssert(dst.shape(3) == nImg);

        if(IsSame<M,dev_memory_space>::Result::value){
            NVMatrix nv_dst    NVView4D(dst);
            NVMatrix nv_delta  NVView4D(delta);
            NVMatrix nv_filter NVView3D(filter);

            /*void convImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,*/
            /*    int imgSize, int paddingStart, int moduleStride, int numImgColors, int numGroups);*/
            convImgActs(nv_delta, nv_filter, nv_dst,
                    nImgPixY, nImgPixX, nModulesY, paddingStart, moduleStride, nImgChan, nGroups,factOld,factNew);
        }else{
            /*void cpuImgActs(float* hidActs, float* filters, float* targets,*/
                           /*int numModulesX,  int numImages,  int numFilters,*/
                           /*int filterSize,  int imgSize,  int moduleStart,*/
                           /*int moduleStride, int numImgColors, int numGroups, bool conv) {*/
            if(factOld == 0.f)
                dst = 0.f;
            cpuImgActs(delta.ptr(), filter.ptr(), dst.ptr(),
                    nModulesX, nImg, nFilt, 
                    sqrt(nFiltPix), nImgPixX, paddingStart,
                    moduleStride, nImgChan, nGroups,true);
        }
    }
template<class V, class M, class L>
	void d_conv2d_dfilt(tensor<V,M,L>& dst_,
			  const tensor<V,M,L>&   delta,
			  const tensor<V,M,L>&   input,
              int paddingStart,
            unsigned int moduleStride, unsigned int nGroups, unsigned int partialSum, float factNew, float factOld){
        if(IsSame<M,host_memory_space>::Result::value){
            std::cout << "warning: host version of d_conv2d_dfilt not implemented"<<std::endl;
            return;
        }

        cuvAssert(dst_.ndim()==3);
        unsigned int nFiltChan = dst_.shape(0);
        unsigned int nFiltPix  = dst_.shape(1);
        unsigned int nFilt     = dst_.shape(2);



        unsigned int filtSize = sqrt(nFiltPix);
        cuvAssert ( nFiltPix == filtSize*filtSize );


        cuvAssert(delta.ndim()==4);
        cuvAssert(delta.shape(0) == nFilt);
        unsigned int nModulesY = delta.shape(1);
        unsigned int nModulesX = delta.shape(2);
        unsigned int nImg      = delta.shape(3);

        cuv::tensor<float,M> dst(extents[(nModulesX*nModulesY)/partialSum][nFiltChan*nFiltPix][nFilt]); // make 3D for NVView3D

        cuvAssert(input.ndim()==4);
        unsigned int nImgChan = input.shape(0);
        unsigned int nImgPixY = input.shape(1);
        unsigned int nImgPixX = input.shape(2);
        cuvAssert(input.shape(3) == nImg);

        /*void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,*/
        /*                    int numModulesX, int filterSize, int paddingStart,*/
        /*                    int moduleStride, int numImgColors, int numGroups, int partialSum);*/
        NVMatrix nv_dst   NVView3D(dst);
        NVMatrix nv_delta NVView4D(delta);
        NVMatrix nv_input NVView4D(input);
        convWeightActs(nv_input, nv_delta, nv_dst,
                nImgPixY, nModulesY,
                nModulesX, filtSize, paddingStart,
                moduleStride, nImgChan, nGroups, partialSum,factOld,factNew);

        dst.reshape(extents[(nModulesX*nModulesY)/partialSum][nFiltChan*nFiltPix*nFilt]);
        dst_.reshape(extents[nFiltChan*nFiltPix*nFilt]);
        cuv::reduce_to_row(dst_,dst);
        dst_.reshape(extents[nFiltChan][nFiltPix][nFilt]);
    }


template<>
    void local_pool(tensor<float,host_memory_space>& target,
            const tensor<float,host_memory_space>& images,
            int subsX, int startX, int strideX, int outputsX, pool_type pooler){
    }
template<>
    void local_pool(tensor<float,dev_memory_space>& target,
            const tensor<float,dev_memory_space>& images,
            int subsX, int startX, int strideX, int outputsX, pool_type pooler){

        cuvAssert(images.ndim()==4);
        unsigned int nFilt    = images.shape(0);
        unsigned int nImgPixY = images.shape(1);
        unsigned int nImgPixX = images.shape(2);
        unsigned int nImg     = images.shape(3);

        cuvAssert(target.ndim()==4);
        cuvAssert(target.shape(0) == nFilt);
        unsigned int nOutPixY = target.shape(1);
        unsigned int nOutPixX = target.shape(2);
        cuvAssert(target.shape(3) == nImg);

        unsigned int poolSize = nImgPixY / nOutPixY;
        cuvAssert(poolSize*nOutPixY == nImgPixY);

        NVMatrix nv_target NVView4D(target);
        NVMatrix nv_images NVView4D(images);
        

        switch(pooler){
            case PT_MAX:
                convLocalPool(nv_images, nv_target, nFilt,
                        subsX, startX, strideX, nOutPixX, MaxPooler());
                break;
            case PT_AVG:
                convLocalPool(nv_images, nv_target, nFilt,
                        subsX, startX, strideX, nOutPixX, AvgPooler(poolSize*poolSize));
                break;
        }
    }
template<>
    void local_max_pool_grad(tensor<float,host_memory_space>& target, const tensor<float,host_memory_space>& images, const tensor<float,host_memory_space>& maxGrads,
            const tensor<float,host_memory_space>& maxActs, int subsX, int startX, int strideX, float factNew,float factOld){
    }
template<>
    void local_max_pool_grad(tensor<float,dev_memory_space>& target, const tensor<float,dev_memory_space>& images, const tensor<float,dev_memory_space>& maxGrads,
            const tensor<float,dev_memory_space>& maxActs, int subsX, int startX, int strideX, float factNew,float factOld){

/*
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * rMaxActs:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 */

        cuvAssert(target.ndim()==4);
        unsigned int nImgChan  = target.shape(0);
        unsigned int nImgPixY  = target.shape(1);
        unsigned int nImgPixX  = target.shape(2);
        unsigned int nImg      = target.shape(3);

        cuvAssert(images.ndim()==4);
        cuvAssert(nImgChan  == images.shape(0));
        cuvAssert(nImgPixY  == images.shape(1));
        cuvAssert(nImgPixX  == images.shape(2));
        cuvAssert(nImg      == images.shape(3));

        cuvAssert(maxGrads.ndim()==4);
        cuvAssert(nImgChan == maxGrads.shape(0));
        unsigned int nOutPixY = maxGrads.shape(1);
        unsigned int nOutPixX = maxGrads.shape(2);
        cuvAssert(nImg     == maxGrads.shape(3));

        cuvAssert(maxActs.ndim()==4);
        cuvAssert(nImgChan == maxActs.shape(0));
        cuvAssert(nOutPixY == maxGrads.shape(1));
        cuvAssert(nOutPixX == maxGrads.shape(2));
        cuvAssert(nImg     == maxActs.shape(3));

        NVMatrix nv_target NVView4D(target);
        NVMatrix nv_images NVView4D(images);
        NVMatrix nv_maxGrads NVView4D(maxGrads);
        NVMatrix nv_maxActs NVView4D(maxActs);
        
/*void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,*/
/*                      int subsX, int startX, int strideX, int outputsX);*/
        convLocalMaxUndo(nv_images,nv_maxGrads, nv_maxActs, nv_target, 
                subsX,startX,strideX,nOutPixX,factOld,factNew);
    }

template<>
    void local_avg_pool_grad(tensor<float,host_memory_space>& target, const tensor<float,host_memory_space>& avgGrads,
            int subsX, int startX, int strideX){
    }
template<>
    void local_avg_pool_grad(tensor<float,dev_memory_space>& target, const tensor<float,dev_memory_space>& avgGrads,
            int subsX, int startX, int strideX){


        cuvAssert(target.ndim()==4);
        unsigned int nImgChan  = target.shape(0);
        unsigned int nImgPixY  = target.shape(1);
        unsigned int nImgPixX  = target.shape(2);
        unsigned int nImg      = target.shape(3);

        cuvAssert(avgGrads.ndim()==4);
        cuvAssert(nImgChan == avgGrads.shape(0));
        unsigned int nOutPixY = avgGrads.shape(1);
        unsigned int nOutPixX = avgGrads.shape(2);
        cuvAssert(nImg == avgGrads.shape(3));

        NVMatrix nv_target NVView4D(target);
        NVMatrix nv_avgGrads NVView4D(avgGrads);
        
        convLocalAvgUndo(nv_avgGrads, nv_target, subsX,startX,strideX,nOutPixX,nImgPixX);
    }

// instantiate
#define  TENS(V,M,T)       tensor<V,M,T>
#define CTENS(V,M,T) const TENS(V,M,T)
#define INST(V,M,T) \
template void reorder_for_conv<V,M,T>(TENS(V,M,T)&, CTENS(V,M,T)&); \
template void reorder_from_conv<V,M,T>(TENS(V,M,T)&, CTENS(V,M,T)&); \
template void convolve2d(TENS(V,M,T)& dst,CTENS(V,M,T)& img,CTENS(V,M,T)& filter, int paddingStart, unsigned int moduleStride, unsigned int nGroups, float factNew, float factOld); \
template void d_conv2d_dfilt(TENS(V,M,T)& dst_, CTENS(V,M,T)& delta, CTENS(V,M,T)&   input, int paddingStart, unsigned int moduleStride, unsigned int nGroups, unsigned int partialSum, float factNew, float factOld);\
template void d_conv2d_dimg(TENS(V,M,T)& dst, CTENS(V,M,T)&   delta, CTENS(V,M,T)&   filter, int paddingStart, unsigned int moduleStride, unsigned int nGroups, float factNew,float factOld);
INST(float,host_memory_space,row_major);
INST(float,dev_memory_space,row_major);
}}

