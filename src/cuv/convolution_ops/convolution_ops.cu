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

#include <boost/scoped_ptr.hpp>
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
#include <cuv/tensor_ops/functors.hpp>

#define NVView1D(X)  \
        (const_cast<float*>(X.ptr()), 1, X.shape(0), X.shape(0), false)
#define NVView3D(X)  \
        (const_cast<float*>(X.ptr()), X.shape(0)*X.shape(1), X.shape(2), X.shape(2),false)
#define NVView4D(X)  \
        (const_cast<float*>(X.ptr()), X.shape(0)*X.shape(1)*X.shape(2), X.shape(3), X.shape(3),false)

// choose maximal thread count depending on cuda arch ( currently the maximum is 1024 for gtx 700 cards )
    
#define MAX_THREADS 512


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
            tensor<int, M, T> colorIndices(extents[nGroups*nFiltChan]);
            sequence(colorIndices);
            convFilterActsSparse(nv_img, nv_filter, nv_dst, colorIndices.ptr(), nImgPixY, nModulesY, nModulesX, paddingStart, moduleStride, nImgChan, nFiltChan, nGroups,factOld,factNew);
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
template<class V, class M, class T>
    void 
    convolve2d(tensor<V,M, T>& dst, 
            const tensor<V,M, T>& img, 
            const tensor<V,M, T>& filter,
            const tensor<int, M, T>& indices,
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

        unsigned int overSample = nGroups * nFiltChan / nImgChan;
        cuvAssert(indices.shape(0) == nGroups);
        /*cuvAssert(indices.shape(1) == nImgChan * nFiltChan);*/
        cuvAssert(indices.shape(1) == overSample * nImgChan);

        // make NVMatrices with this data
        NVMatrix nv_dst    NVView4D(dst);
        NVMatrix nv_img    NVView4D(img);
        NVMatrix nv_filter NVView3D(filter);

        if(IsSame<M,dev_memory_space>::Result::value){
            convFilterActsSparse(nv_img, nv_filter, nv_dst, const_cast<int*>(indices.ptr()),      nImgPixY, nModulesY, nModulesX, paddingStart, moduleStride, nImgChan, nFiltChan, nGroups, factOld,factNew);
        }else{
            throw std::runtime_error("CPU version of convFilterActsSparse not implemented!");
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
                           /*int moduleStride, int numImgColors, int numGroups, bool conv) */
            if(factOld == 0.f)
                dst = 0.f;
            cpuImgActs(delta.ptr(), filter.ptr(), dst.ptr(),
                    nModulesX, nImg, nFilt, 
                    sqrt(nFiltPix), nImgPixX, paddingStart,
                    moduleStride, nImgChan, nGroups,true);
        }
    }
template<class V, class M, class L>
	void d_conv2d_dimg(tensor<V,M,L>& dst,
			  const tensor<V,M,L>&   delta,
			  const tensor<V,M,L>&   filter,
              const tensor<int,M,L>& indices,
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

            /*void convImgActsSparse(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,*/
            /*        int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numFilterColors, int numGroups)*/
            convImgActsSparse(nv_delta, nv_filter, nv_dst, const_cast<int*>(indices.ptr()),
                          nImgPixY,     nImgPixX,       nModulesY,     paddingStart,     moduleStride,         nImgChan, nFiltChan, nGroups, factOld, factNew);
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

        assert((nModulesX * nModulesY) % partialSum == 0);
        cuv::tensor<float,M> dst(extents[(nModulesX*nModulesY)/partialSum][nFiltChan*nFiltPix][nFilt]); // make 3D for NVView3D

        cuvAssert(input.ndim()==4);
        unsigned int nImgChan = input.shape(0);
        unsigned int nImgPixY = input.shape(1);
        unsigned int nImgPixX = input.shape(2);
        cuvAssert(input.shape(3) == nImg);

        /*void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,*/
        /*                    int numModulesX, int filterSize, int paddingStart,*/
        /*                    int moduleStride, int numImgColors, int numGroups, int partialSum);*/
        NVMatrix nv_dst   NVView3D((partialSum > 0 ? dst : dst_));
        NVMatrix nv_delta NVView4D(delta);
        NVMatrix nv_input NVView4D(input);
        convWeightActs(nv_input, nv_delta, nv_dst,
                nImgPixY, nModulesY,
                nModulesX, filtSize, paddingStart,
                moduleStride, nImgChan, nGroups, partialSum);

        if(partialSum > 0){
            dst.reshape(extents[(nModulesX*nModulesY)/partialSum][nFiltChan*nFiltPix*nFilt]);
            dst_.reshape(extents[nFiltChan*nFiltPix*nFilt]);
            cuv::reduce_to_row(dst_,dst, cuv::RF_ADD, factNew, factOld);
            dst_.reshape(extents[nFiltChan][nFiltPix][nFilt]);
        }
    }

template<class V, class M, class L>
	void d_conv2d_dfilt(tensor<V,M,L>& dst_,
			  const tensor<V,M,L>&   delta,
			  const tensor<V,M,L>&   input,
              const tensor<int,M,L>& indices,
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

        boost::scoped_ptr<cuv::tensor<float, M> > dst;
        if(partialSum > 0){
            assert((nModulesX * nModulesY) % partialSum == 0);
            dst.reset(new cuv::tensor<float,M> (extents[(nModulesX*nModulesY)/partialSum][nFiltChan*nFiltPix][nFilt])); // make 3D for NVView3D

            // it seems the current implementation of convWeightActsSparse cannot overwrite memory (?)
            *dst = 0.f;
        }else if(factOld == 0.f){
            dst_ = 0.f;
        }

        cuvAssert(input.ndim()==4);
        unsigned int nImgChan = input.shape(0);
        unsigned int nImgPixY = input.shape(1);
        unsigned int nImgPixX = input.shape(2);
        cuvAssert(input.shape(3) == nImg);

        NVMatrix nv_dst   NVView3D((partialSum > 0 ? *dst : dst_));
        NVMatrix nv_delta NVView4D(delta);
        NVMatrix nv_input NVView4D(input);
/*void convWeightActsSparse(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets, int* dColorIndices,*/
/*                        int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors, int numFilterColors,*/
/*                        int numGroups, int partialSum, float scaleTargets, float scaleOutput) {*/
        convWeightActsSparse(       nv_input,          nv_delta,            nv_dst, const_cast<int*>(indices.ptr()),
                              nImgPixY,       nModulesY,       nModulesX,       filtSize,     paddingStart,     moduleStride,         nImgChan, nFiltChan,
                              nGroups,       partialSum, 
                              partialSum > 0 ? 0 : factOld,   // if partialSum is >0, just overwrite dst and
                              partialSum > 0 ? 1 : factNew);  // care about factNew/factOld below

        if(partialSum > 0){
            dst->reshape(extents[(nModulesX*nModulesY)/partialSum][nFiltChan*nFiltPix*nFilt]);
            dst_.reshape(extents[nFiltChan*nFiltPix*nFilt]);
            cuv::reduce_to_row(dst_,*dst, cuv::RF_ADD, factNew, factOld);
            dst_.reshape(extents[nFiltChan][nFiltPix][nFilt]);
        }
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
        /*cuvAssert(poolSize*nOutPixY == nImgPixY);*/

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
            case PT_SUM:
            convLocalPool(nv_images, nv_target, nFilt,
                    subsX, startX, strideX, nOutPixX, SumPooler());
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


//‘local_sum_pool_grad(cuv::tensor<float, cuv::dev_memory_space, cuv::row_major>&,
//const cuv::tensor<float, cuv::dev_memory_space, cuv::row_major>&,
//const cuv::tensor<float, cuv::dev_memory_space, cuv::row_major>&,
//const cuv::tensor<float, cuv::dev_memory_space, cuv::row_major>&,
//unsigned int&, int, unsigned int&)’


template<>
    void local_sum_pool_grad(tensor<float,host_memory_space>& target, const tensor<float,host_memory_space>& avgGrads,
            int subsX, int startX, int strideX){
    }
template<>
    void local_sum_pool_grad(tensor<float,dev_memory_space>& target, const tensor<float,dev_memory_space>& avgGrads,
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

        convLocalSumUndo(nv_avgGrads, nv_target, subsX,startX,strideX,nOutPixX,nImgPixX);
    }

template<class V, class M, class T>
void response_normalization(tensor<V,M,T>& target, tensor<V,M,T>& denoms, const tensor<V,M,T>& images, int patchSize, float addScale, float powScale){
#ifndef NDEBUG
    if(!images.ndim()==4)
        throw std::runtime_error("response_normalization: images must have dimension 4.");
    if(!target.ndim()==4)
        throw std::runtime_error("response_normalization: target must have dimension 4.");
    if(images.shape()!=target.shape())
        throw std::runtime_error("response_normalization: target must have same shape as images");
    if(denoms.shape()!=target.shape())
        throw std::runtime_error("response_normalization: target must have same shape as denoms");
#endif

    NVMatrix nv_target NVView4D(target);
    NVMatrix nv_denoms NVView4D(denoms);
    NVMatrix nv_images NVView4D(images);
    convResponseNorm(nv_images, nv_denoms, nv_target, target.shape(0), patchSize, addScale, powScale);
}
template<class V, class M, class T>
void response_normalization_grad(tensor<V,M,T>& input_gradients, tensor<V,M,T>& original_outputs, const tensor<V,M,T>& original_inputs,
        const tensor<V,M,T>& delta, const tensor<V,M,T>& denoms, int patchSize, float addScale, float powScale, float factNew, float factOld){
#ifndef NDEBUG
    if(!input_gradients.ndim()==4)
        throw std::runtime_error("response_normalization_grad: input_gradients must have dimension 4.");
    if(!original_outputs.ndim()==4)
        throw std::runtime_error("response_normalization_grad: original_outputs must have dimension 4.");
    if(!original_inputs.ndim()==4)
        throw std::runtime_error("response_normalization_grad: original_inputs must have dimension 4.");
    if(!delta.ndim()==4)
        throw std::runtime_error("response_normalization_grad: delta must have dimension 4.");
    if(!denoms.ndim()==4)
        throw std::runtime_error("response_normalization_grad: denoms must have dimension 4.");
    if(input_gradients.shape() != original_outputs.shape())
        throw std::runtime_error("response_normalization_grad: input_gradients/original_outputs shapes do not match.");
    if(input_gradients.shape() != original_inputs.shape())
        throw std::runtime_error("response_normalization_grad: input_gradients/original_inputs shapes do not match.");
    if(input_gradients.shape() != delta.shape())
        throw std::runtime_error("response_normalization_grad: input_gradients/delta shapes do not match.");
    if(input_gradients.shape() != denoms.shape())
        throw std::runtime_error("response_normalization_grad: input_gradients/denoms shapes do not match.");
#endif

    NVMatrix nv_input_grad NVView4D(input_gradients);
    NVMatrix nv_orig_out NVView4D(original_outputs);
    NVMatrix nv_orig_in  NVView4D(original_inputs);
    NVMatrix nv_delta NVView4D(delta);
    NVMatrix nv_denoms NVView4D(denoms);
    convResponseNormUndo(nv_delta, nv_denoms, nv_orig_in, nv_orig_out, nv_input_grad, input_gradients.shape(0), patchSize, addScale, powScale, factOld, factNew);
}

template<class V, class M, class T>
void contrast_normalization(tensor<V,M,T>& target, tensor<V,M,T>& denoms, const tensor<V,M,T>& meanDiffs, const tensor<V,M,T>& images, int patchSize, float addScale, float powScale){
#ifndef NDEBUG
    if(!images.ndim()==4)
        throw std::runtime_error("response_normalization: images must have dimension 4.");
    if(!target.ndim()==4)
        throw std::runtime_error("response_normalization: target must have dimension 4.");
    if(images.shape()!=target.shape())
        throw std::runtime_error("response_normalization: target must have same shape as images");
    if(denoms.shape()!=target.shape())
        throw std::runtime_error("response_normalization: target must have same shape as denoms");
#endif

    NVMatrix nv_target NVView4D(target);
    NVMatrix nv_denoms NVView4D(denoms);
    NVMatrix nv_meandiffs NVView4D(meanDiffs);
    NVMatrix nv_images NVView4D(images);

    // from layer.cu
    convLocalPool(nv_images, nv_meandiffs, images.shape(0), patchSize, -patchSize/2, 1, images.shape(1), AvgPooler(patchSize*patchSize));
    nv_meandiffs.add(nv_images, -1, 1);
    convContrastNorm(nv_images,nv_meandiffs, nv_denoms,nv_target, target.shape(0), patchSize, addScale, powScale);
}
template<class V, class M, class T>
void contrast_normalization_grad(tensor<V,M,T>& input_gradients, tensor<V,M,T>& original_outputs, const tensor<V,M,T>& meanDiffs, 
        const tensor<V,M,T>& delta, const tensor<V,M,T>& denoms, int patchSize, float addScale, float powScale, float factNew, float factOld){
#ifndef NDEBUG
    if(!input_gradients.ndim()==4)
        throw std::runtime_error("response_normalization_grad: input_gradients must have dimension 4.");
    if(!original_outputs.ndim()==4)
        throw std::runtime_error("response_normalization_grad: original_outputs must have dimension 4.");
    if(!meanDiffs.ndim()==4)
        throw std::runtime_error("response_normalization_grad: meanDiffs must have dimension 4.");
    if(!delta.ndim()==4)
        throw std::runtime_error("response_normalization_grad: delta must have dimension 4.");
    if(!denoms.ndim()==4)
        throw std::runtime_error("response_normalization_grad: denoms must have dimension 4.");
    if(input_gradients.shape() != original_outputs.shape())
        throw std::runtime_error("response_normalization_grad: input_gradients/original_outputs shapes do not match.");
    if(input_gradients.shape() != meanDiffs.shape())
        throw std::runtime_error("response_normalization_grad: input_gradients/meanDiffs shapes do not match.");
    if(input_gradients.shape() != delta.shape())
        throw std::runtime_error("response_normalization_grad: input_gradients/delta shapes do not match.");
    if(input_gradients.shape() != denoms.shape())
        throw std::runtime_error("response_normalization_grad: input_gradients/denoms shapes do not match.");
#endif

    NVMatrix nv_input_grad NVView4D(input_gradients);
    NVMatrix nv_orig_out NVView4D(original_outputs);
    NVMatrix nv_meandiffs  NVView4D(meanDiffs);
    NVMatrix nv_delta NVView4D(delta);
    NVMatrix nv_denoms NVView4D(denoms);
    convContrastNormUndo(nv_delta, nv_denoms, nv_meandiffs, nv_orig_out, nv_input_grad, input_gradients.shape(0), patchSize, addScale, powScale, factOld, factNew);

    // "spread" delta from above according to mean
    convLocalAvgUndo(nv_delta, nv_orig_out, patchSize, -patchSize/2, 1, input_gradients.shape(1), input_gradients.shape(1));
    nv_input_grad.add(nv_orig_out, 1,-1);
}

template<class V, class M, class T>
void gaussian_blur(tensor<V,M,T>& target, const tensor<V,M,T>& images, const tensor<V,M,T>& filter, bool horiz, float factNew, float factOld){
#ifndef NDEBUG
    if(!target.ndim()==4)
        throw std::runtime_error("gaussian_blur: target must have dimension 4.");
    if(!images.ndim()==4)
        throw std::runtime_error("gaussian_blur: images must have dimension 4.");
    if(filter.ndim()!=1)
        throw std::runtime_error("gaussian_blur: filter must have dimension 1.");
    if(filter.size()==1 || (((filter.size() - 1) / 2) * 2 + 1 != filter.size()))
        throw std::runtime_error("gaussian_blur: filter must have size 2*k+1.");
    if(target.shape() != images.shape())
        throw std::runtime_error("gaussian_blur: images and targets must have same shape.");
#endif
    NVMatrix nv_images NVView4D(images);
    NVMatrix nv_target NVView4D(target);
    NVMatrix nv_filter NVView1D(filter);
    convGaussianBlur(nv_images, nv_filter, nv_target, horiz, images.shape(0), factOld, factNew);
}

template<class V, class M, class T>
void bed_of_nails(tensor<V,M,T>& target, const tensor<V,M,T>& images, int startX, int strideX, float factNew, float factOld){
#ifndef NDEBUG
    if(!target.ndim()==4)
        throw std::runtime_error("bed_of_nails: target must have dimension 4.");
    if(!images.ndim()==4)
        throw std::runtime_error("bed_of_nails: images must have dimension 4.");
    if(target.shape(1) != images.shape(1) / strideX)
        throw std::runtime_error("bed_of_nails: images and targets shapes must relate by strideX.");
#endif
    NVMatrix nv_images NVView4D(images);
    NVMatrix nv_target NVView4D(target);
    convBedOfNails(nv_images, nv_target, images.shape(0), images.shape(1), startX, strideX, factOld, factNew);
}

template<class V, class M, class T>
void bed_of_nails_grad(tensor<V,M,T>& target, const tensor<V,M,T>& delta, int startX, int strideX, float factNew, float factOld){
#ifndef NDEBUG
    if(!target.ndim()==4)
        throw std::runtime_error("bed_of_nails_grad: target must have dimension 4.");
    if(!delta.ndim()==4)
        throw std::runtime_error("bed_of_nails_grad: delta must have dimension 4.");
    if(delta.shape(1) != target.shape(1) / strideX)
        throw std::runtime_error("bed_of_nails_grad: deltas and targets shapes must relate by strideX.");
#endif
    NVMatrix nv_delta NVView4D(delta);
    NVMatrix nv_target NVView4D(target);
    convBedOfNailsUndo(nv_delta, nv_target, target.shape(0), target.shape(1), startX, strideX, factOld, factNew);
}

template<class V, class M, class T>
void crop(tensor<V,M,T>& cropped, const tensor<V,M,T>& images, int startY, int startX){
    NVMatrix nv_cropped NVView4D(cropped);
    NVMatrix nv_images NVView4D(images);

    convCrop(nv_images, nv_cropped, images.shape(1), cropped.shape(1), startY, startX);
}

template<class V, class M, class T>
void project_to_ball(tensor<V,M,T>& filters, float ball){
    if(filters.ndim() == 3){
        // n_modules = 1
        NVMatrix nv_filters NVView3D(filters);
        normalizeLocalWeights(nv_filters, 1, ball);
    }else if(filters.ndim() == 4){
        NVMatrix nv_filters NVView4D(filters);
        normalizeLocalWeights(nv_filters, filters.shape(0), ball);
    }else{
        throw std::runtime_error("project_to_ball: don't know how to normalize your supplied filters. dim!=3 and dim!=4");
    }
}

template<class V, class M, class T>
void resize_bilinear(tensor<V,M,T>& dest, const tensor<V,M,T>& images, float scale){
    NVMatrix nv_dest NVView4D(dest);
    NVMatrix nv_images NVView4D(images);

    convResizeBilinear(nv_images, nv_dest, images.shape(1), dest.shape(1), scale);
}

template<class V, class M, class T>
void response_norm_cross_map(tensor<V,M,T>& target, tensor<V,M,T>& denoms, const tensor<V,M,T>& images, int sizeF, float addScale, float powScale, bool blocked){
#ifndef NDEBUG
    if(!images.ndim()==4)
        throw std::runtime_error("response_norm_cross_map: images must have dimension 4.");
    if(!target.ndim()==4)
        throw std::runtime_error("response_norm_cross_map: target must have dimension 4.");
    if(images.shape()!=target.shape())
        throw std::runtime_error("response_norm_cross_map: target must have same shape as images");
    if(denoms.shape()!=target.shape())
        throw std::runtime_error("response_norm_cross_map: target must have same shape as denoms");
#endif

    NVMatrix nv_target NVView4D(target);
    NVMatrix nv_denoms NVView4D(denoms);
    NVMatrix nv_images NVView4D(images);
    convResponseNormCrossMap(nv_images,nv_denoms,nv_target, target.shape(0), sizeF, addScale, powScale, blocked);
}

template<class V, class M, class T>
void response_norm_cross_map_grad(tensor<V,M,T>& input_gradients, tensor<V,M,T>& original_outputs, const tensor<V,M,T>& original_inputs, 
        const tensor<V,M,T>& delta, const tensor<V,M,T>& denoms, int sizeF, float addScale, float powScale, bool blocked, float factNew, float factOld){
#ifndef NDEBUG
    if(!input_gradients.ndim()==4)
        throw std::runtime_error("response_norm_cross_map_grad: input_gradients must have dimension 4.");
    if(!original_outputs.ndim()==4)
        throw std::runtime_error("response_norm_cross_map_grad: original_outputs must have dimension 4.");
    if(!original_inputs.ndim()==4)
        throw std::runtime_error("response_norm_cross_map_grad: original_inputs must have dimension 4.");
    if(!delta.ndim()==4)
        throw std::runtime_error("response_norm_cross_map_grad: delta must have dimension 4.");
    if(!denoms.ndim()==4)
        throw std::runtime_error("response_norm_cross_map_grad: denoms must have dimension 4.");
    if(input_gradients.shape() != original_outputs.shape())
        throw std::runtime_error("response_norm_cross_map_grad: input_gradients/original_outputs shapes do not match.");
    if(input_gradients.shape() != original_inputs.shape())
        throw std::runtime_error("response_norm_cross_map_grad: input_gradients/original_inputs shapes do not match.");
    if(input_gradients.shape() != delta.shape())
        throw std::runtime_error("response_norm_cross_map_grad: input_gradients/delta shapes do not match.");
    if(input_gradients.shape() != denoms.shape())
        throw std::runtime_error("response_norm_cross_map_grad: input_gradients/denoms shapes do not match.");
#endif

    NVMatrix nv_input_grad NVView4D(input_gradients);
    NVMatrix nv_orig_out NVView4D(original_outputs);
    NVMatrix nv_orig_in  NVView4D(original_inputs);
    NVMatrix nv_delta NVView4D(delta);
    NVMatrix nv_denoms NVView4D(denoms);
    convResponseNormCrossMapUndo(nv_delta,nv_denoms,nv_orig_in, nv_orig_out, nv_input_grad, input_gradients.shape(0), sizeF, addScale, powScale, blocked, factOld, factNew);
}




template<bool FirstDim, tuplewise_op_functor to,class T>
__global__
void tuplewise_op_kernel(T* dst, const T* src, unsigned int dst_rows, unsigned int dst_cols, unsigned int subspace_size, float eps){
    if(FirstDim){
        unsigned int line = blockIdx.x;
        T* dst0 = dst + line * dst_cols;
        const T* src_ptr = src + (subspace_size * line) * dst_cols;

      for(unsigned int item = threadIdx.x; item < dst_cols; item += blockDim.x){
                T squared_sum = 0.f;
                unsigned int end = item + subspace_size * dst_cols;
                for (unsigned int index = item; index <  end; index+=dst_cols){
                    T s = src_ptr[index];
                    switch(to){
                        case TO_NORM:
                        case TO_ADD_SQUARED:
                            squared_sum += s * s;
                            break;
                        case TO_MAX:
                            squared_sum = max(s, squared_sum);
                            break;
                        case TO_SUBSAMPLE:
                            if(index == item)
                                squared_sum = src_ptr[index];
                            break;
                        case TO_MEAN:
                            squared_sum += src_ptr[index];
                            break;
                    }
                }
                if(to == TO_NORM)
                    dst0[item] = sqrt(squared_sum + eps);
                else if (to == TO_MEAN)
                    dst0[item] = squared_sum / subspace_size;
                else
                    dst0[item] = squared_sum;
      }
    }else{
        unsigned int item = blockIdx.x;
        unsigned int line = threadIdx.x;
        
        T* dst0 = dst + item * dst_rows;
        const T* src_ptr = src + (subspace_size*dst_rows * item);

        for(; line < dst_rows; line += blockDim.x){
            T squared_sum =  0.f;
            unsigned int end =  subspace_size*(line+1);
            unsigned int begin = subspace_size*line;
            for (unsigned int index = begin; index < end; index++){
                T s = src_ptr[index];
                switch(to){
                    case TO_NORM:
                    case TO_ADD_SQUARED:
                        squared_sum += s * s;
                        break;
                    case TO_MAX:
                        squared_sum = max(s, squared_sum);
                        break;
                    case TO_SUBSAMPLE:
                        if(index == begin)
                            squared_sum = src_ptr[index];
                        break;
                    case TO_MEAN:
                        squared_sum += src_ptr[index];
                        break;
                }
            }
            if(to == TO_NORM)
                dst0[line] = sqrt(squared_sum + eps);
            else if (to == TO_MEAN)
                dst0[line] = squared_sum / subspace_size;
            else
                dst0[line] = squared_sum;
        }
    }
}

template<bool FirstDim, tuplewise_op_functor to, class T>
__global__
void tuplewise_op_grad_kernel(T* dst, const T* src, const T* delta, unsigned int dst_rows, unsigned int dst_cols, unsigned int subspace_size, float eps){
    if(FirstDim){
        unsigned int line = blockIdx.x;
        const T* src_ptr = src + (subspace_size * line) * dst_cols;
        T* dst_ptr = dst + (subspace_size * line) * dst_cols;
        const T* d0  = delta + line * dst_cols;
        T p;

        for(unsigned int item = threadIdx.x; item < dst_cols; item += blockDim.x){        
                // calculates squared sum
                float squared_sum = 0.f; 
                unsigned int max_index = 0;
                unsigned int end = item + subspace_size * dst_cols;
                for (unsigned int index = item; index < end; index += dst_cols){
                    T s = src_ptr[index];
                    switch(to){
                        case TO_NORM:
                        case TO_ADD_SQUARED:
                            squared_sum += s*s;
                            break;
                        case TO_MAX:
                            if (s > squared_sum){
                                squared_sum  = s;
                                max_index = index;
                            }
                            break;
                    }
                }
                
                switch(to){
                    case TO_NORM:
                        p  = d0[item] / (sqrt(squared_sum+eps));
                        break;
                    case TO_MAX:
                        p  = d0[item];
                        break;
                    case TO_ADD_SQUARED:
                        p  = 2.f * d0[item];
                        break;
                    case TO_SUBSAMPLE:
                        p  = d0[item];
                        break;
                    case TO_MEAN:
                        p  = d0[item] * (1.f / subspace_size);
                        break;
                };
                

                // updates dst for each feature in subspace 
                for (unsigned int index = item; index < end; index+= dst_cols){
                    switch(to){
                        case TO_NORM:
                            dst_ptr[index] = p * src_ptr[index];
                            break;
                        case TO_MAX:
                            if (max_index == index)
                                dst_ptr[index] = p;
                            else 
                                dst_ptr[index] = 0.f;
                            break;
                        case TO_ADD_SQUARED:
                            dst_ptr[index] = p * src_ptr[index];
                            break;
                        case TO_SUBSAMPLE:
                            if (index == item)
                                dst_ptr[index] = p;
                            else 
                                dst_ptr[index] = 0.f;
                            break;
                        case TO_MEAN:
                            dst_ptr[index] = p;
                            break;
                    }
                }
        }
    }else{
        unsigned int item = blockIdx.x;
        unsigned int line = threadIdx.x;
        const T* src_ptr = src + (item * subspace_size*dst_rows);
        T* dst_ptr = dst + (item * subspace_size*dst_rows);
        const T* d0  = delta + item * dst_rows;
        T p;

        for(; line < dst_rows; line += blockDim.x){
            float squared_sum = 0.f;
            unsigned int max_index = 0;
            unsigned int end = subspace_size*(line+1);

            for (unsigned int index = subspace_size*line; index < end; index++){
                T s = src_ptr[index];
                switch(to){
                    case TO_NORM:
                    case TO_ADD_SQUARED:
                        squared_sum += s*s;
                        break;
                    case TO_MAX:
                        if (s > squared_sum){
                            squared_sum  =  s;
                            max_index = index;
                        }
                        break;
                }
            }

            switch(to){
                case TO_NORM:
                    p  = d0[line] / (sqrt(squared_sum+eps));
                    break;
                case TO_MAX:
                    p = d0[line];
                    break;
                case TO_ADD_SQUARED:
                    p = 2.f * d0[line];
                    break;
                case TO_SUBSAMPLE:
                    p  = d0[line];
                    break;
                case TO_MEAN:
                    p  = d0[line] * (1.f / subspace_size);
                    break;
            }

            unsigned int begin_idx = subspace_size*line;
            for (unsigned int index = begin_idx; index < end; index++){
                switch(to){
                    case TO_NORM:
                        dst_ptr[index] = p * src_ptr[index];
                        break;
                    case TO_MAX:
                        if (max_index == index)
                            dst_ptr[index] = p;
                        else 
                            dst_ptr[index] = 0;
                        break;
                    case TO_ADD_SQUARED:
                        dst_ptr[index] = p * src_ptr[index];
                        break;
                    case TO_SUBSAMPLE:
                        if (index == begin_idx)
                            dst_ptr[index] = p;
                        else 
                            dst_ptr[index] = 0.f;
                        break;
                    case TO_MEAN:
                        dst_ptr[index] = p;
                        break;
                }
            }
        }
    }
}





template<bool FirstDim, tuplewise_op_functor to, class T>
    void tuplewise_op_host(T* dst, const T* src, unsigned int lines, unsigned int items, unsigned int subspace_size, float eps){
        if(FirstDim){
            for(unsigned int line = 0; line < lines; line++){
                T* dst_ptr = dst + line * items;
                const T* src_ptr = src + (subspace_size * line) * items;

                for(unsigned int i=0; i < items; i++){
                    float squared_sum = 0.f;
                    for (unsigned int index = i; index < i + subspace_size * items; index += items){
                        switch(to){
                            case TO_NORM:
                            case TO_ADD_SQUARED:
                                squared_sum += src_ptr[index] * src_ptr[index];
                                break;
                            case TO_MAX:
                                squared_sum = max(src_ptr[index], squared_sum);
                                break;
                            case TO_SUBSAMPLE:
                                if(index == i)
                                    squared_sum = src_ptr[index];
                                break;
                            case TO_MEAN:
                                squared_sum += src_ptr[index];
                                break;

                        }
                    }

                    if(to == TO_NORM)
                        dst_ptr[i] = sqrt(squared_sum + eps);
                    else if (to == TO_MEAN)
                        dst_ptr[i] = squared_sum / subspace_size;
                    else
                        dst_ptr[i] = squared_sum;
                }
            }
        }else{
            for(unsigned int item = 0; item < items; item++){
                T* dst_ptr = dst + item * lines;
                const T* src_ptr = src + (item * subspace_size * lines);
                for(unsigned int i = 0; i < lines; i++){
                    float squared_sum = 0.f;
                    for (unsigned int index = subspace_size*i; index < subspace_size*(i+1); index++){
                        switch(to){
                            case TO_NORM:
                            case TO_ADD_SQUARED:
                                squared_sum += src_ptr[index] * src_ptr[index];
                                break;
                            case TO_MAX:
                                squared_sum = max(src_ptr[index], squared_sum);
                                break;
                            case TO_SUBSAMPLE:
                                if(index == subspace_size*i)
                                    squared_sum = src_ptr[index];
                                break;
                            case TO_MEAN:
                                squared_sum += src_ptr[index];
                                break;
                        }
                    }
                    if(to == TO_NORM)
                        dst_ptr[i] = sqrt(squared_sum + eps);
                    else if (to == TO_MEAN)
                        dst_ptr[i] = squared_sum / subspace_size;
                    else
                        dst_ptr[i] = squared_sum;
                }
            }

        }

    }


template<class V,class M, class T>
    void tuplewise_op(tensor<V,M,T>& dst, const tensor<V,M,T>& src, unsigned int dim, unsigned int subspace_size, tuplewise_op_functor to, float eps){
        assert(dim == 0 || dim == src.ndim()-1);
        unsigned int items = dst.size() / dst.shape(dim);
        unsigned int lines = dst.shape(dim);

        cuvAssert(dst.shape(dim)==src.shape(dim)/subspace_size);
        cuvAssert(src.shape(dim) % subspace_size == 0);

        


        if(IsSame<M,host_memory_space>::Result::value){
            switch(to){
                case TO_NORM:
                    if(dim == 0){
                        tuplewise_op_host<true, TO_NORM>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);
                    }else{
                        tuplewise_op_host<false, TO_NORM>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);
                    }
                    break;
                case TO_MAX:
                    if(dim == 0){
                        tuplewise_op_host<true, TO_MAX>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);

                    }else{
                        tuplewise_op_host<false, TO_MAX>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);
                    }
                    break;
                case TO_ADD_SQUARED:
                    if(dim == 0){
                        tuplewise_op_host<true, TO_ADD_SQUARED>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);
                    }else{
                        tuplewise_op_host<false, TO_ADD_SQUARED>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);
                    }
                    break;
                case TO_SUBSAMPLE:
                    if(dim == 0){
                        tuplewise_op_host<true, TO_SUBSAMPLE>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);
                    }else{
                        tuplewise_op_host<false, TO_SUBSAMPLE>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);
                    }
                    break;
                case TO_MEAN:
                    if(dim == 0){
                        tuplewise_op_host<true, TO_MEAN>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);
                    }else{
                        tuplewise_op_host<false, TO_MEAN>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);
                    }
                    break;
            }
        }else{
            // device: run kernel
            unsigned int num_blocks  = lines;
            unsigned int num_threads = min(512, int(32 * ceil( items / 32. )));
            if(dim != 0){
              num_threads =  min(512, int(32 * ceil( lines / 32. )));
              num_blocks  = items;
            }
                    
            switch(to){
                case TO_NORM:
                    if(dim == 0){
                        tuplewise_op_kernel<true, TO_NORM><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);
                    }else{
                        tuplewise_op_kernel<false, TO_NORM><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);

                    }
                    break;
                case TO_MAX:
                    if(dim == 0){
                        tuplewise_op_kernel<true, TO_MAX><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);
                    }else{
                        tuplewise_op_kernel<false, TO_MAX><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);

                    }
                    break;
                case TO_ADD_SQUARED:
                    if(dim == 0){
                        tuplewise_op_kernel<true, TO_ADD_SQUARED><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);
                    }else{
                        tuplewise_op_kernel<false, TO_ADD_SQUARED><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);

                    }
                    break;
                case TO_SUBSAMPLE:
                    if(dim == 0){
                        tuplewise_op_kernel<true, TO_SUBSAMPLE><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);
                    }else{
                        tuplewise_op_kernel<false, TO_SUBSAMPLE><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);

                    }
                    break;
                case TO_MEAN:
                    if(dim == 0){
                        tuplewise_op_kernel<true, TO_MEAN><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);
                    }else{
                        tuplewise_op_kernel<false, TO_MEAN><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), lines, items, subspace_size, eps);

                    }
                    break;
            }
            cuvSafeCall(cudaThreadSynchronize());
        }
    }


template<bool FirstDim, tuplewise_op_functor to,class T>
void tuplewise_op_grad_host(T* dst, const T* src, const T* delta, unsigned int lines, unsigned int items, unsigned int subspace_size, float eps){
    if(FirstDim){
        for(unsigned int line = 0; line < lines; line++){
            const T* d_ptr  = delta + line * items;
            const T* src_ptr = src + (subspace_size * line) * items;
            T* dst_ptr = dst + (subspace_size * line) * items;
            for(unsigned int i=0; i < items; i++){
                float squared_sum = 0;
                unsigned int max_index = 0;
                // calculates squared sum
                for (unsigned int index = i; index < i + subspace_size * items; index+= items){
                    switch(to){
                        case TO_NORM:
                            squared_sum += src_ptr[index] * src_ptr[index];
                            break;
                        case TO_MAX:
                            if (src_ptr[index] > squared_sum){
                                squared_sum  =  src_ptr[index];
                                max_index = index;
                            }
                            break;
                    }
                }

                float f;
                switch(to){
                    case TO_NORM:
                        f = d_ptr[i] / (sqrt(squared_sum + eps));
                        break;
                    case TO_MAX:
                        f = d_ptr[i];
                        break;
                    case TO_ADD_SQUARED:
                        f = 2.f * d_ptr[i];
                        break;
                    case TO_SUBSAMPLE:
                        f = d_ptr[i];
                        break;
                    case TO_MEAN:
                        f = d_ptr[i] * (1.f / subspace_size);
                        break;
                };
                // updates dst for each feature in subspace 
                for (unsigned int index = i; index < i + subspace_size * items; index+= items){
                    switch(to){
                        case TO_NORM:
                            dst_ptr[index] = f * src_ptr[index];
                            break;
                        case TO_MAX:
                            if (max_index == index){
                                dst_ptr[index] = f;
                            }
                            else{ 
                                dst_ptr[index] = 0;
                            }
                            break;
                        case TO_ADD_SQUARED:
                            dst_ptr[index] = f *  src_ptr[index];
                            break;
                        case TO_SUBSAMPLE:
                            if(index == i)
                                dst_ptr[index] = f;
                            else
                                dst_ptr[index] = 0.f;
                            break;
                        case TO_MEAN:
                            dst_ptr[index] = f;
                            break;
                    }
                }
            }
        }
    }else{
        for(unsigned int item = 0; item < items; item++){
            const T* src_ptr = src + (item * subspace_size * lines);

            T* dst_ptr = dst + (item * subspace_size * lines);
            const T* d_ptr  = delta + item * lines;
            for(unsigned int i=0; i < lines; i++){
                float squared_sum = 0.f;
                unsigned int max_index = 0;
                unsigned int end = subspace_size*(i+1);
                for (unsigned int index = subspace_size*i; index < end; index++){
                    switch(to){
                        case TO_NORM:
                            squared_sum += src_ptr[index] * src_ptr[index];
                            break;
                        case TO_MAX:
                            if (src_ptr[index] > squared_sum){
                                squared_sum  =  src_ptr[index];
                                max_index = index;
                            }
                            break;
                    }
                }

                float f;
                switch(to){
                    case TO_NORM:
                        f = d_ptr[i] / (sqrt(squared_sum+eps));
                        break;
                    case TO_MAX:
                        f = d_ptr[i];
                        break;
                    case TO_ADD_SQUARED:
                        f = 2.f * d_ptr[i];
                        break;
                    case TO_SUBSAMPLE:
                        f = d_ptr[i];
                        break;
                    case TO_MEAN:
                        f = d_ptr[i] * (1.f / subspace_size);
                        break;
                };

                for (unsigned int index = subspace_size*i; index < end; index++){
                    switch(to){
                        case TO_NORM:
                            dst_ptr[index] = f * src_ptr[index];
                            break;
                        case TO_MAX:
                            if (max_index == index)
                                dst_ptr[index] = f;
                            else 
                                dst_ptr[index] = 0;
                            break;
                        case TO_ADD_SQUARED:
                            dst_ptr[index] = f * src_ptr[index];
                            break;
                        case TO_SUBSAMPLE:
                            if(index == subspace_size * i)
                                dst_ptr[index] = f;
                            else
                                dst_ptr[index] = 0.f;
                            break;
                        case TO_MEAN:
                            dst_ptr[index] = f;
                            break;
                    }
                }
            }
        }
    }
}


template<class V,class M, class T>
    void tuplewise_op_grad(tensor<V,M,T>& dst, const tensor<V,M,T>& src, const tensor<V,M,T>& delta, unsigned int dim, unsigned int subspace_size, tuplewise_op_functor to, float eps){
        assert(dim == 0 || dim == src.ndim()-1);
        assert(dst.shape()==src.shape());
        cuvAssert(delta.shape(dim)==src.shape(dim)/subspace_size);

        unsigned int items = delta.size() / delta.shape(dim);
        unsigned int lines = delta.shape(dim);
        if(IsSame<M,host_memory_space>::Result::value){
            switch(to){
                case TO_NORM:
                    if(dim == 0){
                        tuplewise_op_grad_host<true, TO_NORM>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);
                    }else{
                        tuplewise_op_grad_host<false, TO_NORM>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);
                    }
                    break;
                case TO_MAX:
                    if(dim == 0){
                        tuplewise_op_grad_host<true, TO_MAX>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);

                    }else{
                        tuplewise_op_grad_host<false, TO_MAX>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);
                    }
                    break;
                case TO_ADD_SQUARED:
                    if(dim == 0){
                        tuplewise_op_grad_host<true, TO_ADD_SQUARED>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);

                    }else{
                        tuplewise_op_grad_host<false, TO_ADD_SQUARED>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);
                    }
                    break;
                case TO_SUBSAMPLE:
                    if(dim == 0){
                        tuplewise_op_grad_host<true, TO_SUBSAMPLE>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);

                    }else{
                        tuplewise_op_grad_host<false, TO_SUBSAMPLE>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);
                    }
                    break;
                case TO_MEAN:
                    if(dim == 0){
                        tuplewise_op_grad_host<true, TO_MEAN>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);

                    }else{
                        tuplewise_op_grad_host<false, TO_MEAN>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);
                    }
                    break;
            }
        }else{
            // device: run kernel
            unsigned int num_blocks  = lines;
            unsigned int num_threads = min(512, int(32 * ceil( items / 32. )));
            if(dim != 0){
              num_threads =  min(512, int(32 * ceil( lines / 32. )));
              num_blocks  = items;
            }

            switch(to){
                case TO_NORM:
                    if(dim == 0){
                        tuplewise_op_grad_kernel<true, TO_NORM><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);
                    }else{
                        tuplewise_op_grad_kernel<false, TO_NORM><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);
                    }
                    break;
                case TO_MAX:
                    if(dim == 0){
                        tuplewise_op_grad_kernel<true, TO_MAX><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);

                    }else{
                        tuplewise_op_grad_kernel<false, TO_MAX><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);
                    }
                    break;
                case TO_ADD_SQUARED:
                    if(dim == 0){
                        tuplewise_op_grad_kernel<true, TO_ADD_SQUARED><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);

                    }else{
                        tuplewise_op_grad_kernel<false, TO_ADD_SQUARED><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);
                    }
                    break;
                case TO_SUBSAMPLE:
                    if(dim == 0){
                        tuplewise_op_grad_kernel<true, TO_SUBSAMPLE><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);

                    }else{
                        tuplewise_op_grad_kernel<false, TO_SUBSAMPLE><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);
                    }
                    break;
                case TO_MEAN:
                    if(dim == 0){
                        tuplewise_op_grad_kernel<true, TO_MEAN><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);

                    }else{
                        tuplewise_op_grad_kernel<false, TO_MEAN><<<num_blocks,num_threads>>>(dst.ptr(), src.ptr(), delta.ptr(),  lines, items, subspace_size, eps);
                    }
                    break;
            }

            cuvSafeCall(cudaThreadSynchronize());
        }
    }

/********************************************************************************************************
 * overlapping weighted sub tensor op start
 *****************************************************************************************************************/
/// logarithm of the sum of exponentiations of the inputs in a numerically stable way. log(exp(x)+exp(y))

template<weighted_sub_tensor_op_functor to, class T>
__global__
void weighted_sub_tensor_op_kernel(T* dst, unsigned char* dst_max_idx, const T* src, const T* m_W,
        unsigned int dst_rows, unsigned int dst_cols,  unsigned int dst_colsx, unsigned int dst_colsy, unsigned int src_size,
        unsigned int stride, unsigned int subspace_size, float eps){
        unsigned int line =  blockIdx.x;
        T* dst0 = dst  + line * dst_cols;
        unsigned char * dst_max_idx0;

        unsigned int shift = stride * line;
        unsigned int last_idx = shift + subspace_size;
        unsigned int diff;     
        //check boundaries
        extern __shared__ float w[] ;

        const T* m_W_ptr = m_W + line * subspace_size; 
        for (unsigned int i = threadIdx.x; i < subspace_size; i += blockDim.x)
              w[i] = m_W_ptr[i];
        
        if (last_idx > src_size){
             diff = (subspace_size - (last_idx - src_size))* dst_cols;
        } else {
            diff = subspace_size * dst_cols;
        }
        __syncthreads();

        switch(to){
            case TO_WMAX:
            case TO_WMAX_LOGSPACE:
                dst_max_idx0 = dst_max_idx   + line * dst_cols;
                break;
        }

        const T* src_ptr = src + shift * dst_cols; 

        T squared_sum;
        T temp;
        unsigned int end ;
        unsigned int wInd ;
        unsigned char max_idx;
        bf_logaddexp<float> lae;
        for(unsigned int item = threadIdx.x; item < dst_cols; item += blockDim.x){
                squared_sum = 0.f;
                end = item + diff;
                wInd = 0;
                
                for (unsigned int index = item; index < end; index += dst_cols, wInd++){            
                    T s = src_ptr[index];
                    switch(to){
                        case TO_LOGWADDEXP:
                            squared_sum = lae(squared_sum, w[wInd]*s);
                            break;
                        case TO_LOGWADDEXP_LOGSPACE:
                            squared_sum = lae(squared_sum, w[wInd] + s); 
                            break;
                        case TO_WADD:
                            squared_sum +=  w[wInd] * s;
                            break;
                        case TO_WMAX:
                            temp =  w[wInd] * s;
                            if (temp > squared_sum){
                                squared_sum = temp;
                                max_idx = wInd;
                                }
                            break;
                        case TO_WMAX_LOGSPACE:
                            temp =  w[wInd] + s;
                            if (temp > squared_sum){
                                    squared_sum = temp;
                                    max_idx = wInd;
                               }
                            break;
                        }
            }
            __syncthreads();
       switch(to){
          case TO_WMAX:
          case TO_WMAX_LOGSPACE:
              dst0[item] = squared_sum;
              dst_max_idx0[item] = max_idx;
              break;
          case TO_LOGWADDEXP:
              dst0[item] = squared_sum + eps;
              break;
          case TO_LOGWADDEXP_LOGSPACE:
              dst0[item] = squared_sum + eps;
              break;
          case TO_WADD:
              dst0[item] = squared_sum;
              break;
            }
        }
}


template<weighted_sub_tensor_op_functor to, class T>
    void weighted_sub_tensor_op_host(T* dst, unsigned char* dst_max_idx, const T* src, const T* m_W,
                    unsigned int lines, unsigned int items, unsigned int src_size,
                    unsigned int stride, unsigned int subspace_size, float eps){
                    
            unsigned char * dst_max_idx0;
            unsigned int diff;
            
            unsigned int global_buffer_size = 32;
            float squared_sum[global_buffer_size];
            
            for(unsigned int line = 0; line < lines; line++){
                    T* dst0 = dst  + line * items;
                    unsigned int buffer_size = global_buffer_size;

                    unsigned int shift = stride * line;
                    unsigned int last_idx = shift + subspace_size;
     
                    //check boundaries

                    const T* m_W_ptr = m_W + line * subspace_size;       
                    if (last_idx > src_size){
                        diff = (subspace_size - (last_idx - src_size))* items;
                    } else {
                        diff = subspace_size * items;
                    }

                    switch(to){
                        case TO_WMAX:
                        case TO_WMAX_LOGSPACE:
                            dst_max_idx0 = dst_max_idx   + line * items;
                            break;
                    }

                    const T* src_ptr = src + shift * items; 

                    T temp;
                    unsigned int end ;
                    unsigned int wInd ;
                    unsigned char max_idx;
                    bf_logaddexp<float> lae;
                    unsigned int item;
                    for(item = 0; item < items; item += buffer_size){
                            //reset buffer
                            //for ( unsigned int i = 0; i < buffer_size; i++) squared_sum[i] = 0.f;
			    memset(squared_sum, 0, buffer_size * sizeof(float));
                            
                            end = item + diff;
                            wInd = 0;
                            
                            //check if we can use the whole buffer
                            if ((item + buffer_size) > items) buffer_size = items - item; 
                            
                            for (unsigned int index = item; index < end; index += items, wInd++){   
                                for ( unsigned int buff = 0; buff < buffer_size; buff ++){
                                T s = src_ptr[index + buff];
                                switch(to){
                                    case TO_LOGWADDEXP:
                                        squared_sum[buff] = lae(squared_sum[buff], m_W_ptr[wInd]*s);
                                        break;
                                    case TO_LOGWADDEXP_LOGSPACE:
                                        squared_sum[buff] = lae(squared_sum[buff], m_W_ptr[wInd] + s); 
                                        break;
                                    case TO_WADD:
                                        squared_sum[buff] +=  m_W_ptr[wInd] * s;
                                        break;
                                    case TO_WMAX:
                                        temp =  m_W_ptr[wInd] * s;
                                        if (temp > squared_sum[buff]){
                                            squared_sum[buff] = temp;
                                            max_idx = wInd;
                                            }
                                        break;
                                    case TO_WMAX_LOGSPACE:
                                        temp =  m_W_ptr[wInd] + s;
                                        if (temp > squared_sum[buff]){
                                                squared_sum[buff] = temp;
                                                max_idx = wInd;
                                        }
                                        break;
                                    }
                                    
                            }//enf of for buffer
                        }
                //write whole buffer        
                for ( unsigned int buff = 0; buff < buffer_size; buff ++){        
                    switch(to){
                        case TO_WMAX:
                        case TO_WMAX_LOGSPACE:
                            dst0[item + buff] = squared_sum[buff];
                            dst_max_idx0[item + buff] = max_idx;
                            break;
                        case TO_LOGWADDEXP:
                            dst0[item + buff] = squared_sum[buff] + eps;
                            break;
                        case TO_LOGWADDEXP_LOGSPACE:
                            dst0[item + buff] = squared_sum[buff] + eps;
                            break;
                        case TO_WADD:
                            dst0[item + buff] = squared_sum[buff];
                            break;
                            }
                        }
                    }           
            }
}


template<class V,class M, class T>
    void weighted_sub_tensor_op(tensor<V,M,T>& dst, tensor<unsigned char,M,T>& dst_max_idx, const tensor<V,M,T>& src, const tensor<V,M,T>& m_W, unsigned int size, unsigned int stride, unsigned int subspace_size, weighted_sub_tensor_op_functor to, float eps){
        
        unsigned int itemx;
        unsigned int itemy;
        cuvAssert(src.shape().size() == 3 || src.shape().size() == 4);
        //calculate number of items, based on dimension of image
        if (dst.shape().size() == 4){
            itemx = dst.shape(1) * dst.shape(2);
            itemy = dst.shape(3);
        } else {
            itemx = dst.shape(1);
            itemy = dst.shape(2);            
        }
            unsigned int lines = dst.shape(0);
            unsigned int items = src.size() / src.shape(0);

        cuvAssert(subspace_size <= 256);
        cuvAssert(dst.shape(0)==size);
        //check dimensions of dst
        cuvAssert(dst.shape(0)*stride == src.shape(0));

        unsigned int src_size = src.shape(0);
    
        if(IsSame<M,host_memory_space>::Result::value){
            switch(to){
                case TO_WMAX:
                        weighted_sub_tensor_op_host<TO_WMAX>(dst.ptr(), dst_max_idx.ptr(), src.ptr(), m_W.ptr(), lines, items, src_size, stride, subspace_size, eps);
                    break;
                case TO_WMAX_LOGSPACE:
                        weighted_sub_tensor_op_host<TO_WMAX_LOGSPACE>(dst.ptr(), dst_max_idx.ptr(), src.ptr(), m_W.ptr(), lines, items, src_size, stride, subspace_size, eps);
                    break;
                case TO_LOGWADDEXP:
                        weighted_sub_tensor_op_host<TO_LOGWADDEXP>(dst.ptr(), dst_max_idx.ptr(), src.ptr(), m_W.ptr(), lines, items, src_size, stride, subspace_size, eps);
                    break;
                case TO_LOGWADDEXP_LOGSPACE:
                        weighted_sub_tensor_op_host<TO_LOGWADDEXP_LOGSPACE>(dst.ptr(), dst_max_idx.ptr(), src.ptr(), m_W.ptr(), lines, items, src_size, stride, subspace_size, eps);
                    break;
                case TO_WADD:
                        weighted_sub_tensor_op_host<TO_WADD>(dst.ptr(), dst_max_idx.ptr(), src.ptr(), m_W.ptr(), lines, items, src_size, stride, subspace_size, eps);
                    break;
        }
        }else{
            // device: run kernel
            unsigned int num_blocks  = lines;
            // in order to stay sync, each block must calculate at least one batch
            unsigned int num_threads = min(MAX_THREADS, int(32 * ceil( items / 32. )));
            unsigned int sharedMemory  = (subspace_size)*sizeof(V);

            switch(to){
                case TO_WMAX:
                        weighted_sub_tensor_op_kernel<TO_WMAX><<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), dst_max_idx.ptr(), src.ptr(), m_W.ptr(), lines, items, itemx, itemy, src_size,  stride, subspace_size, eps);
                    break;
                case TO_WMAX_LOGSPACE:
                        weighted_sub_tensor_op_kernel<TO_WMAX_LOGSPACE><<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), dst_max_idx.ptr(), src.ptr(), m_W.ptr(), lines, items,  itemx, itemy, src_size,  stride, subspace_size, eps);
                    break;
                case TO_LOGWADDEXP:
                        weighted_sub_tensor_op_kernel<TO_LOGWADDEXP><<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), dst_max_idx.ptr(), src.ptr(), m_W.ptr(), lines, items,  itemx, itemy, src_size, stride, subspace_size, eps);
                    break;
                case TO_LOGWADDEXP_LOGSPACE:
                        weighted_sub_tensor_op_kernel<TO_LOGWADDEXP_LOGSPACE><<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), dst_max_idx.ptr(), src.ptr(), m_W.ptr(), lines, items, itemx, itemy, src_size, stride, subspace_size, eps);
                    break;          
                case TO_WADD:
                        weighted_sub_tensor_op_kernel<TO_WADD><<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), dst_max_idx.ptr(), src.ptr(), m_W.ptr(), lines, items, itemx, itemy, src_size, stride, subspace_size, eps);
                    break;

            }
            cuvSafeCall(cudaThreadSynchronize());
        }
    }


/***************************************************************
*  weighted_subTensor_op grad implementation
* 
 ****************************************************************/

/************************************
 * helper functions to be able to calculate atomic add float in test mode with old GPU
 * 
 * code from https://www.sharcnet.ca/help/index.php/CUDA_tips_and_tricks
 * 
 * Atomic ops may block concurrent threads, however all atomic ops will be performed.
 * This is not a problem here, because there are no reads on dst.
 * Further the conflicting write operations, which are solved using atomics, 
 * are very few compared to the complete work of a thread.
 * 
 * sources for atomics
 * http://www.nvidia.de/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf
 * atomic ops kepler ( since GTX Titan and gtx 680 have the same chip..
 * http://www.geforce.com/Active/en_US/en_US/pdf/GeForce-GTX-680-Whitepaper-FINAL.pdf
 * http://www.nvidia.de/content/PDF/kepler/NVIDIA-Kepler-GK110-Architecture-Whitepaper.pdf
 *******************************/
 __device__ inline void atomic_Add(float* address, float value)
    {

   #if __CUDA_ARCH__ >= 200
      atomicAdd(address,value);
   #else
   int oldval, newval, readback;

   oldval = __float_as_int(*address);
   newval = __float_as_int(__int_as_float(oldval) + value);
   while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval)
     {
      oldval = readback;
      newval = __float_as_int(__int_as_float(oldval) + value);
     }
   #endif
   }
   


 
template<bool spn, weighted_sub_tensor_op_functor to, class T>
__global__
void weighted_sub_tensor_op_grad_kernel(T* dst, T* w_delta, const T* src, const T* delta, const T* m_W, const T* r0, const T* S, const unsigned char* max_idx, const bool d_dx, const bool d_dw,
        unsigned int dst_rows, unsigned int dst_colsx, unsigned int dst_colsy, unsigned int src_size,
        unsigned int size, unsigned int stride, unsigned int subspace_size, float eps){
    if(spn){
        unsigned int line = blockIdx.x;
        unsigned int shift = stride * line;
        
        // shared array to store intermediate results for weight derivative
        extern __shared__ T rw[];
        T* res_w = &rw[0] + threadIdx.y * blockDim.x + subspace_size;

        //init shared memory of this thead
        res_w[threadIdx.x] = 0;

        extern __shared__ float w[];
        const T* m_W_ptr = m_W + line * subspace_size;
        float* w_ptr = &w[0] + subspace_size + blockDim.x * blockDim.y;
 
        for (unsigned int i = threadIdx.x; i < subspace_size; i += blockDim.x)
              w_ptr[i] = m_W_ptr[i];

        extern __shared__ T sum[];
 	    for (unsigned int s = threadIdx.y * blockDim.x + threadIdx.x; s < subspace_size; s += blockDim.x * blockDim.y) sum[s] = 0;	
        __syncthreads();
    
        unsigned int dst_cols = dst_colsx *dst_colsy;
        unsigned int shift_t_d = shift * dst_cols;
        unsigned int line_t_sub = line * subspace_size;
        unsigned int line_t_d = line * dst_cols;
        unsigned int last_idx = shift + subspace_size;

        unsigned int diff;     
        //check boundaries
        if (last_idx > src_size){
             diff = (subspace_size - (last_idx - src_size))* dst_cols;
        } else {
            diff = subspace_size * dst_cols;
        }

        const T* src_ptr = src + shift_t_d;
        T* dst_ptr       = dst + shift_t_d;

        const T* r0_ptr;
        const T* d0  =     delta     + line_t_d; // für alle elemente
        const unsigned char* max_idx_ptr;

        switch(to){
           case TO_WMAX:
           case TO_WMAX_LOGSPACE:
                max_idx_ptr =     max_idx    + line_t_d;
                break;
           case TO_LOGWADDEXP:
           case TO_LOGWADDEXP_LOGSPACE:
               r0_ptr = r0 + line_t_d;
               break;
        }

        T* dw_ptr     = w_delta +  line_t_sub;
        T p;
        T S_val;
        
        for(unsigned int itemy = threadIdx.y; itemy < dst_colsy; itemy += blockDim.y){
            //weight gradient in case SOFT_INFEERENCE 
            switch(to){
               case TO_LOGWADDEXP:
                   S_val = 1/(S[itemy] +  eps);
               case TO_LOGWADDEXP_LOGSPACE:
                   S_val = 1/(S[itemy] + eps);
            }
            for(unsigned int itemx = threadIdx.x; itemx < dst_colsx; itemx += blockDim.x){
                unsigned int item = itemy * dst_colsx + itemx;
                unsigned char maxIdx;
                switch(to){
                    case TO_WMAX:
                    case TO_WMAX_LOGSPACE:
                        maxIdx = max_idx_ptr[item];
                }
 
            switch(to){
               case TO_WMAX:
               case TO_WADD:
               case TO_WMAX_LOGSPACE:
                    //get derivative from parent node..
                    p  = d0[item];
                    break;
               case TO_LOGWADDEXP:
                   p  = d0[item] * 1/expf(r0_ptr[item]);
               case TO_LOGWADDEXP_LOGSPACE:
                   p  = d0[item] * 1/expf(r0_ptr[item]);
                   break;
            }

            unsigned int end = item + diff;
            unsigned int wInd = 0;

            // updates dst for each feature in subspace
            __syncthreads();
            for (unsigned int index = item; index < end; index+= dst_cols, wInd++){
                float temp = 0;
                T src_val = src_ptr[index];
                switch(to){
                    case TO_WMAX:
                        if (maxIdx == wInd){
                            if(d_dx) atomic_Add (&dst_ptr[index], p * w_ptr[wInd]);
                            if(d_dw) res_w[threadIdx.x]  = 1;
                        }break;
                    case TO_WMAX_LOGSPACE:
                        if (maxIdx == wInd){
                            if(d_dx) atomic_Add(&dst_ptr[index], p);
                            if(d_dw) res_w[threadIdx.x]  = 1;
                        }break;
                    case TO_LOGWADDEXP:
                        temp = expf(w_ptr[wInd] * src_val) * p;
                        if(d_dx) atomic_Add(&dst_ptr[index],  w_ptr[wInd] * temp);
                        if(d_dw) res_w[threadIdx.x] =  S_val * src_val * temp;
                        break;
                    case TO_LOGWADDEXP_LOGSPACE:
                        temp = expf(src_val + w_ptr[wInd]) * p;
                        if(d_dx) atomic_Add(&dst_ptr[index],  temp);
                        if(d_dw) res_w[threadIdx.x]  = S_val * temp;
                        break;
                    case TO_WADD:
                        if(d_dx) atomic_Add(&dst_ptr[index], p*w_ptr[wInd]);
                        if(d_dw) res_w[threadIdx.x]  = p*src_val;
                        break;
                }

                if (d_dw){
                    //reduction for threadIdx.x
                    for ( unsigned int i = blockDim.x/2; i > 0; i/=2){
                            __syncthreads();
                            if (threadIdx.x < i ){
                                res_w[threadIdx.x] += res_w[threadIdx.x + i];
                            }
                        }

                    //now there it should be reduced to a line
                    if (threadIdx.x == 0)
                        for ( unsigned int j = blockDim.y/2; j > 0; j/=2){
                            __syncthreads();
                                    if (threadIdx.y < j){
                                            res_w[threadIdx.x] += rw [subspace_size + (threadIdx.y + j) * blockDim.x];
                                    }
                        }

                       __syncthreads();
                            if ((threadIdx.x == 0) && (threadIdx.y == 0)){
                                	    sum[wInd]  += res_w[threadIdx.x];
                            }
		        	//init shared memory of this thead
      				res_w[threadIdx.x] = 0;
                }
            }
            }
        }
            //write result to global memory
            if (d_dw){
                __syncthreads();
                for (unsigned int update_idx = threadIdx.y * blockDim.x + threadIdx.x; update_idx < subspace_size; update_idx += blockDim.x*blockDim.y){
                 dw_ptr[update_idx] += sum[update_idx];
                }
            }
    }else{
        unsigned int line = blockIdx.x;
        unsigned int shift = stride * line;
        
        extern __shared__ T rw[];
        T* res_w = &rw[0] + threadIdx.y * blockDim.x + subspace_size;

        //init shared memory of this thead
        res_w[threadIdx.x] = 0;

        extern __shared__ float w[];
        const T* m_W_ptr = m_W + line * subspace_size;
        float* w_ptr = &w[0] + subspace_size + blockDim.x * blockDim.y;
 
        for (unsigned int i = threadIdx.x; i < subspace_size; i += blockDim.x)
              w_ptr[i] = m_W_ptr[i];

        extern __shared__ T sum[];
            for (unsigned int s = threadIdx.y * blockDim.x + threadIdx.x; s < subspace_size; s += blockDim.x * blockDim.y) sum[s] = 0;
        __syncthreads();


        unsigned int dst_cols = dst_colsx *dst_colsy;
        unsigned int shift_t_d = shift * dst_cols;
        unsigned int line_t_sub = line * subspace_size;
        unsigned int line_t_d = line * dst_cols;
        unsigned int last_idx = shift + subspace_size;

        unsigned int diff;
        
        //check boundaries
        if (last_idx > src_size){
             diff = (subspace_size - (last_idx - src_size))* dst_cols;
        } else {
            diff = subspace_size * dst_cols;
        }

        const T* src_ptr = src + shift_t_d;
        T* dst_ptr       = dst + shift_t_d;

        const T* r0_ptr;
        const T* d0  =     delta     + line_t_d; // für alle elemente
        const unsigned char* max_idx_ptr;

        switch(to){
           case TO_WMAX:
           case TO_WMAX_LOGSPACE:
                max_idx_ptr =     max_idx    + line_t_d;
                break;
           case TO_LOGWADDEXP:
           case TO_LOGWADDEXP_LOGSPACE:
               r0_ptr = r0 + line_t_d;
               break;
        }

        T* dw_ptr     = w_delta +  line_t_sub;
        T p;


        for(unsigned int itemy = threadIdx.y; itemy < dst_colsy; itemy += blockDim.y){
            for(unsigned int itemx = threadIdx.x; itemx < dst_colsx; itemx += blockDim.x){
                unsigned int item = itemy * dst_colsx + itemx;
                unsigned char maxIdx;
                switch(to){
                    case TO_WMAX:
                    case TO_WMAX_LOGSPACE:
                        maxIdx = max_idx_ptr[item];
                }


            switch(to){
               case TO_WMAX:
               case TO_WADD:
               case TO_WMAX_LOGSPACE:
                    //get derivative from parent node..
                    p  = d0[item];
                    break;
               case TO_LOGWADDEXP:
                   p  = d0[item] * 1/expf(r0_ptr[item]);
               case TO_LOGWADDEXP_LOGSPACE:
                   p  = d0[item] * 1/expf(r0_ptr[item]);
                   break;
            }

            unsigned int end = item + diff;
            unsigned int wInd = 0;

            //check boundaries
            __syncthreads();
            // updates dst for each feature in subspace
            for (unsigned int index = item; index < end; index+= dst_cols, wInd++){
                float temp = 0;
                T s = src_ptr[index];
                switch(to){
                    case TO_WMAX:
                        if (maxIdx == wInd){
                            if(d_dx) atomic_Add (&dst_ptr[index], p * w_ptr[wInd]);
                            if(d_dw) res_w[threadIdx.x]  = p * s;
                        }break;
                    case TO_WMAX_LOGSPACE:
                        if (maxIdx == wInd){
                            if(d_dx) atomic_Add(&dst_ptr[index], p);
                            if(d_dw) res_w[threadIdx.x]  = p;
                        }break;
                    case TO_LOGWADDEXP:
                        temp = expf( w_ptr[wInd] * s) * p;
                        if(d_dx) atomic_Add(&dst_ptr[index],  w_ptr[wInd] * temp);
                        if(d_dw) res_w[threadIdx.x] =  s * temp;
                        break;
                    case TO_LOGWADDEXP_LOGSPACE:
                        temp = expf( s + w_ptr[wInd] ) * p;
                        if(d_dx) atomic_Add(&dst_ptr[index],temp);
                        if(d_dw) res_w[threadIdx.x]  = temp;
                        break;
                    case TO_WADD:
                        if(d_dx) atomic_Add(&dst_ptr[index], p*w_ptr[wInd]);
                        if(d_dw) res_w[threadIdx.x]  = p*s;
                        break;
                }

                if (d_dw){
                    //reduction for threadIdx.x
                    for ( unsigned int i = blockDim.x/2; i > 0; i/=2){
                            __syncthreads();
                            if (threadIdx.x < i ){
                                res_w[threadIdx.x] += res_w[threadIdx.x + i];
                            }
                        }

                    //now there it should be reduced to a line
                    if (threadIdx.x == 0)
                        for ( unsigned int j = blockDim.y/2; j > 0; j/=2){
                            __syncthreads();
                                    if (threadIdx.y < j){
                                            res_w[threadIdx.x] += rw [subspace_size + (threadIdx.y + j) * blockDim.x];
                                    }
                        }

                            if ((threadIdx.x == 0) && (threadIdx.y == 0)){
                                            sum[wInd]  += res_w[threadIdx.x];
                            }

                       __syncthreads();
                       //init shared memory of this thead
                        res_w[threadIdx.x] = 0;
                }
            }
            }
        }

            //write result to global memory
            if (d_dw){
                __syncthreads();
                for (unsigned int update_idx = threadIdx.y * blockDim.x + threadIdx.x; update_idx < subspace_size; update_idx += blockDim.x*blockDim.y){
                 dw_ptr[update_idx] += sum[update_idx];
                }
            }
    }
}


template<bool spn, weighted_sub_tensor_op_functor to, class T>
void weighted_sub_tensor_op_grad_host(T* dst, T* w_delta, const T* src, const T* delta, const T* m_W, const T* r0, const T* S, const unsigned char* max_idx, const bool d_dx, const bool d_dw,
                     unsigned int lines, unsigned int dst_colsx, unsigned int dst_colsy, unsigned int src_size, unsigned int size, unsigned int stride, unsigned int subspace_size, float eps){
       if(spn){
        unsigned int dst_cols = dst_colsx *dst_colsy;
        unsigned int diff;     
 
        for ( unsigned int line = 0; line < lines; line ++){
            unsigned int shift = stride * line;

            const T* m_W_ptr = m_W + line * subspace_size;
    
            unsigned int shift_t_d = shift * dst_cols;
            unsigned int line_t_sub = line * subspace_size;
            unsigned int line_t_d = line * dst_cols;
            unsigned int last_idx = shift + subspace_size;
            //check boundaries
            if (last_idx > src_size){
                diff = (subspace_size - (last_idx - src_size))* dst_cols;
            } else {
                diff = subspace_size * dst_cols;
            }

            const T* src_ptr = src + shift_t_d;
            T* dst_ptr       = dst + shift_t_d;

            const T* r0_ptr;
            const T* d0  =     delta     + line_t_d; // für alle elemente
            const unsigned char* max_idx_ptr;

            switch(to){
            case TO_WMAX:
            case TO_WMAX_LOGSPACE:
                    max_idx_ptr =     max_idx    + line_t_d;
                    break;
            case TO_LOGWADDEXP:
            case TO_LOGWADDEXP_LOGSPACE:
                r0_ptr = r0 + line_t_d;
                break;
            }

            T* dw_ptr     = w_delta +  line_t_sub;
            T p;
            T S_val;
            for(unsigned int itemy = 0; itemy < dst_colsy; itemy ++){
                //weight gradient in case SOFT_INFEERENCE 
                switch(to){
                case TO_LOGWADDEXP:
                case TO_LOGWADDEXP_LOGSPACE:
                    S_val = 1/(S[itemy]+ eps);
                }
                for(unsigned int itemx = 0; itemx < dst_colsx; itemx ++){
                    unsigned int item = itemy * dst_colsx + itemx;
                    unsigned char maxIdx;
                    switch(to){
                        case TO_WMAX:
                        case TO_WMAX_LOGSPACE:
                            maxIdx = max_idx_ptr[item];
                    }
    
                switch(to){
                case TO_WMAX:
                case TO_WADD:
                case TO_WMAX_LOGSPACE:
                        //get derivative from parent node..
                        p  = d0[item];
                        break;
                case TO_LOGWADDEXP:
                case TO_LOGWADDEXP_LOGSPACE:
                    p  = d0[item] * 1/expf(r0_ptr[item]);
                    break;
                }

                unsigned int end = item + diff;
                unsigned int wInd = 0;

                // updates dst for each feature in subspace
                for (unsigned int index = item; index < end; index+= dst_cols, wInd++){
                    float temp = 0;
                    T src_val = src_ptr[index];
                    switch(to){
                        case TO_WMAX:
                            if (maxIdx == wInd){
                                if(d_dx) dst_ptr[index]+= p * m_W_ptr[wInd];
                                if(d_dw) dw_ptr[wInd]  += 1;
                            }break;
                        case TO_WMAX_LOGSPACE:
                            if (maxIdx == wInd){
                                if(d_dx) dst_ptr[index]+= p;
                                if(d_dw) dw_ptr[wInd]  += 1;
                            }break;
                        case TO_LOGWADDEXP:
                            temp = expf(m_W_ptr[wInd] * src_val) * p;
                            if(d_dx) dst_ptr[index]+=  m_W_ptr[wInd] * temp;
                            if(d_dw) dw_ptr[wInd] +=  S_val * src_val * temp;
                            break;
                        case TO_LOGWADDEXP_LOGSPACE:
                            temp = expf(src_val + m_W_ptr[wInd] ) * p;
                            if(d_dx) dst_ptr[index] +=temp;
                            if(d_dw) dw_ptr[wInd]  += S_val * temp;
                            break;
                        case TO_WADD:
                            if(d_dx) dst_ptr[index] += p*m_W_ptr[wInd];
                            if(d_dw) dw_ptr[wInd]  += p*src_val;
                            break;
                    }
                }
                }
            }
        }
       }else {
        unsigned int dst_cols = dst_colsx *dst_colsy;
        unsigned int diff;     
 
        for ( unsigned int line = 0; line < lines; line ++){
        unsigned int shift = stride * line;

        const T* m_W_ptr = m_W + line * subspace_size;
   
        unsigned int shift_t_d = shift * dst_cols;
        unsigned int line_t_sub = line * subspace_size;
        unsigned int line_t_d = line * dst_cols;
        unsigned int last_idx = shift + subspace_size;
        //check boundaries
        if (last_idx > src_size){
             diff = (subspace_size - (last_idx - src_size))* dst_cols;
        } else {
            diff = subspace_size * dst_cols;
        }

        const T* src_ptr = src + shift_t_d;
        T* dst_ptr       = dst + shift_t_d;

        const T* r0_ptr;
        const T* d0  =     delta     + line_t_d; // für alle elemente
        const unsigned char* max_idx_ptr;

        switch(to){
           case TO_WMAX:
           case TO_WMAX_LOGSPACE:
                max_idx_ptr =     max_idx    + line_t_d;
                break;
           case TO_LOGWADDEXP:
           case TO_LOGWADDEXP_LOGSPACE:
               r0_ptr = r0 + line_t_d;
               break;
        }

        T* dw_ptr     = w_delta +  line_t_sub;
        T p;
        for(unsigned int item = 0; item < dst_cols; item ++){
                unsigned char maxIdx;
                switch(to){
                    case TO_WMAX:
                    case TO_WMAX_LOGSPACE:
                        maxIdx = max_idx_ptr[item];
                }
 
            switch(to){
               case TO_WMAX:
               case TO_WADD:
               case TO_WMAX_LOGSPACE:
                    //get derivative from parent node..
                    p  = d0[item];
                    break;
               case TO_LOGWADDEXP:
               case TO_LOGWADDEXP_LOGSPACE:
                   p  = d0[item] * 1/expf(r0_ptr[item]);
                   break;
            }

            unsigned int end = item + diff;
            unsigned int wInd = 0;

            // updates dst for each feature in subspace
            for (unsigned int index = item; index < end; index+= dst_cols, wInd++){
                float temp = 0;
                T src_val = src_ptr[index];
                switch(to){
                    case TO_WMAX:
                        if (maxIdx == wInd){
                            if(d_dx) dst_ptr[index]+= p * m_W_ptr[wInd];
                            if(d_dw) dw_ptr[wInd]  += p * src_val;
                        }break;
                    case TO_WMAX_LOGSPACE:
                        if (maxIdx == wInd){
                            if(d_dx) dst_ptr[index]+= p;
                            if(d_dw) dw_ptr[wInd]  += p;
                        }break;
                    case TO_LOGWADDEXP:
                        temp = expf(m_W_ptr[wInd] * src_val) * p;
                        if(d_dx) dst_ptr[index]+=  m_W_ptr[wInd] * temp;
                        if(d_dw) dw_ptr[wInd] +=   src_val * temp;
                        break;
                    case TO_LOGWADDEXP_LOGSPACE:
                        temp = expf(src_val + m_W_ptr[wInd]) * p;
                        if(d_dx) dst_ptr[index] += temp;
                        if(d_dw) dw_ptr[wInd]   += temp;
                        break;
                    case TO_WADD:
                        if(d_dx) dst_ptr[index] += p*m_W_ptr[wInd];
                        if(d_dw) dw_ptr[wInd]  += p*src_val;
                        break;
                }
            }
        }
      }
    }
}




template<class V,class M, class T>
    void weighted_sub_tensor_op_grad(tensor<V,M,T>& dst, tensor<V,M,T>& w_delta, const tensor<V,M,T>& src, const tensor<V,M,T>& delta, const tensor<V,M,T>& m_W, const tensor<V,M,T>& r0, 
                                     const tensor<V,M,T>& S, const tensor< unsigned char,M,T>& max_idx, const bool spn, const bool d_dx, const bool d_dw, unsigned int size, unsigned int stride, 
                                     unsigned int subspace_size, weighted_sub_tensor_op_functor to, float eps){
        assert(dst.shape()==src.shape());
        assert(w_delta.shape()==m_W.shape());
        assert(w_delta.shape(0) == delta.shape(0));
        assert(m_W.shape(0) == delta.shape(0));
        assert(m_W.shape(1)==subspace_size);
        assert(w_delta.shape(1)==subspace_size);    
        assert(delta.shape(0) == src.shape(0)/stride);
        assert(subspace_size <= 256); // (data type char is used to store max_idx)
        cuvAssert(delta.shape().size() == 3 || delta.shape().size() == 4);
    
        //initialize  w_delta and dst
        cuv::fill (dst, 0);
        cuv::fill (w_delta, 0);

        unsigned int src_size = src.shape(0);        
        unsigned int lines = delta.shape(0);
        unsigned int itemx;
        unsigned int itemy;
        //calculate number of items, based on dimension of image
        if (delta.shape().size() == 4){
            itemx = delta.shape(1) * delta.shape(2);
            itemy = delta.shape(3);
        } else {
            itemx = delta.shape(1);
            itemy = delta.shape(2);            
        }
        
        if(IsSame<M,host_memory_space>::Result::value){
            switch(to){
                case TO_WMAX:
                    if (spn) weighted_sub_tensor_op_grad_host<true,  TO_WMAX>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                    else     weighted_sub_tensor_op_grad_host<false, TO_WMAX>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                    break;
                case TO_WMAX_LOGSPACE:
                    if (spn) weighted_sub_tensor_op_grad_host<true,  TO_WMAX_LOGSPACE>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                    else     weighted_sub_tensor_op_grad_host<false, TO_WMAX_LOGSPACE>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                    break;
                case TO_LOGWADDEXP:
                    if (spn) weighted_sub_tensor_op_grad_host<true,  TO_LOGWADDEXP>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                    else     weighted_sub_tensor_op_grad_host<false, TO_LOGWADDEXP>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                    break;
                case TO_LOGWADDEXP_LOGSPACE:
                    if (spn) weighted_sub_tensor_op_grad_host<true,  TO_LOGWADDEXP_LOGSPACE>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                    else     weighted_sub_tensor_op_grad_host<false, TO_LOGWADDEXP_LOGSPACE>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                    break;          
                case TO_WADD:
                    if (spn) weighted_sub_tensor_op_grad_host<true,  TO_WADD>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                    else     weighted_sub_tensor_op_grad_host<false, TO_WADD>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                    break;
            }
    }else{
            // device: run kernel        
            //define size of the block
            unsigned int block_size = 16;
            dim3 num_threads;

            num_threads.x = block_size;
            num_threads.y = block_size;
    
            unsigned int num_blocks  = lines;
            unsigned int sharedMemory  = (2*subspace_size + (block_size * block_size))*sizeof(V);

            switch(to){
                case TO_WMAX:
                        if(spn) weighted_sub_tensor_op_grad_kernel<true , TO_WMAX><<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                        else    weighted_sub_tensor_op_grad_kernel<false, TO_WMAX><<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                        break;
                case TO_WMAX_LOGSPACE:
                        if(spn) weighted_sub_tensor_op_grad_kernel<true , TO_WMAX_LOGSPACE><<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                        else    weighted_sub_tensor_op_grad_kernel<false, TO_WMAX_LOGSPACE><<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                    break;

                case TO_LOGWADDEXP:
                        if(spn) weighted_sub_tensor_op_grad_kernel<true , TO_LOGWADDEXP><<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                        else    weighted_sub_tensor_op_grad_kernel<false, TO_LOGWADDEXP><<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                    break;
                case TO_LOGWADDEXP_LOGSPACE:
                        if(spn) weighted_sub_tensor_op_grad_kernel<true , TO_LOGWADDEXP_LOGSPACE><<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                        else    weighted_sub_tensor_op_grad_kernel<false, TO_LOGWADDEXP_LOGSPACE><<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                    break;
                case TO_WADD:
                        if(spn) weighted_sub_tensor_op_grad_kernel<true , TO_WADD><<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                        else    weighted_sub_tensor_op_grad_kernel<false, TO_WADD><<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), w_delta.ptr(), src.ptr(), delta.ptr(), m_W.ptr(), r0.ptr(), S.ptr(), max_idx.ptr(), d_dx, d_dw, lines, itemx, itemy, src_size, size, stride, subspace_size, eps);
                    break;
            }

            cuvSafeCall(cudaThreadSynchronize());
        }
    }  
  

/*****************************************************************************************************************
 * overlapping tuplewise op end
 *****************************************************************************************************************/

/*****************************************************************************************************************
 * spn_output_op
 *****************************************************************************************************************/




template<class T>
__global__
void spn_output_op_kernel(T* dst, const T* src, const T* m_W, const T* Y, unsigned int lines, unsigned int items, unsigned int batch){       
        //load weights into shared memory
        float result = 0;
        unsigned int itb = (items * batch);
        extern __shared__ float w[];
        const T* Y_ptr = Y + threadIdx.x * lines;
        int y;
        bf_logaddexp<float> lae;
        
        for (unsigned int i = threadIdx.x; i < lines; i += blockDim.x) {
           w[i] = m_W[i];
        }
        __syncthreads();
        for ( unsigned int b = threadIdx.x; b < batch; b += blockDim.x, Y_ptr += blockDim.x*lines){
            y = int( Y_ptr[0] );
            
            for ( unsigned int x = 0; x < items; x++){
                unsigned int  xtb = x * batch;
                T* dst_ptr = dst + xtb;
                // marginalization step? then we have to calculate all values
                if (y < 0){
                    for ( unsigned int c = 0; c < lines; c++){
                        const T* src_ptr = src + c * itb + xtb;
                        result = lae(result, w[c] + src_ptr[b]);
                    }
                } else {
                    // no marginalization step => all labels =! y were 0
                    const T* src_ptr = src + y * itb + xtb;
                    result = lae(result, w[y] + src_ptr[b]);                  
                }
                dst_ptr[b] = result;
                result = 0;
            }
        }
}



template<class T>
    void spn_output_op_host(T* dst, const T* src, const T* m_W, const T* Y, unsigned int lines, unsigned int items, unsigned int batch){
        unsigned int itb = (items * batch);
        bf_logaddexp<float> lae;
        for ( unsigned int b = 0; b < batch; b ++){
            const T* Y_ptr = Y + b*lines;
            for ( unsigned int x = 0; x <items; x++){
                unsigned int  xtb = x * batch;                
                T* dst_ptr = dst +  xtb;
                int y = int( Y_ptr[0] );
                if ( y < 0){
                    for ( unsigned int c = 0; c < lines; c++){
                        const T* src_ptr = src + c * itb + xtb;
                        dst_ptr[x] = lae(dst_ptr[x], m_W[c] + src_ptr[b]);       
                    }
                } else {
                    const T* src_ptr = src + y * itb + xtb;
                    dst_ptr[x] = lae(dst_ptr[x], m_W[y] + src_ptr[b]);       
                }
            }
        }     
}


template<class V, class M, class T>
void spn_output_op(tensor<V,M,T>& dst, const tensor<V,M,T>& src, const tensor<V,M,T>& m_W, const tensor<V,M,T>& Y){
        unsigned int lines = src.shape(0);
        unsigned int items;
        unsigned int batch;
        cuvAssert((dst.shape().size() == 3) || (dst.shape().size() == 4));
        if ( dst.shape().size() > 3 ){ 
            items = src.shape(1) * src.shape(2);
            batch = src.shape(3);
        } else {
            items = src.shape(1);
            batch = src.shape(2);
        }
        
        //check every param for nans
        cuvAssert(!cuv::has_nan(src));
        cuvAssert(!cuv::has_nan(m_W));
        cuvAssert(!cuv::has_nan(Y));
        
        cuvAssert(m_W.ndim() == 1);
        cuvAssert((src.ndim() == 3) || src.ndim() == 4);
        cuvAssert(Y.ndim() == 2);
        
        cuvAssert(lines > 1);
        cuvAssert(items > 0);
        cuvAssert(batch > 0);        
        cuvAssert(src.shape(0) ==  Y.shape(1));
        cuvAssert( (src.shape(2) == Y.shape(0)) || (src.shape(3) == Y.shape(0)));
        cuvAssert(dst.shape(0) == 1);
        cuvAssert(dst.shape(1) ==  src.shape(1));
        cuvAssert(dst.shape(2) ==  src.shape(2));
        if ( dst.shape().size() > 3 ) cuvAssert(dst.shape(3) ==  src.shape(3));
        cuvAssert(m_W.shape(0) ==  src.shape(0));
        
        if(IsSame<M,host_memory_space>::Result::value){
                spn_output_op_host(dst.ptr(), src.ptr(), m_W.ptr(), Y.ptr(), lines, items, batch);
        }else{
            // device: run kernel
            unsigned int num_blocks  = 1;
            unsigned int num_threads = min(MAX_THREADS, int(32 * ceil( batch / 32. )));
            unsigned int sharedMemory  = (lines)*sizeof(float);
                       
            spn_output_op_kernel<<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), src.ptr(), m_W.ptr(), Y.ptr(), lines, items, batch);
           
            cuvSafeCall(cudaThreadSynchronize());
        }
   }
    
    //TODO ALLE ABLEITUNGEN NOCH MAL CHECKEN
    /*****************************************************************************************************************************
     * spn_output_op grad
     *******************************************************************************************************************************/
// currently d_dy just works for case x = 1, 
template<class T>
__global__
void spn_output_op_grad_kernel(T* dst, const T* src,  T* w_delta, T* Y_delta, const T* m_W,  const T* Y, const T* S, const T* lae_res, const T* delta,
                          unsigned int lines, unsigned int items, unsigned int batch, const bool d_dx, const bool d_dy, const bool d_dw, float eps){       
        extern __shared__ float temp_w_delta[];
        unsigned int itb = (items * batch);
        float s = 0;
        int y;
        temp_w_delta[threadIdx.x] = 0;       

        //each block calculates derivatives for one class        
        int c = blockIdx.x;
        
        //load weight into shared memory
        float w = m_W[c];
        for ( unsigned int b = threadIdx.x; b < batch; b += blockDim.x){
            unsigned int btl = b * lines;
            const T* Y_ptr = Y + btl;
            T* Y_delta_ptr = Y_delta + btl;
            //get correct label (or marginalization flag)
            y = int( Y_ptr[0] );
            T s_val = 1/(S[b] + eps);
            
            for ( unsigned int x = 0; x < items; x++){
                unsigned int xtb = x * batch;
                const T* delta_ptr = delta + xtb;
                const T* lae_ptr = lae_res + xtb; //1 / exp(lae_res[b]);  
                const T exp_res = 1 / expf(lae_ptr[b]);
                    //set derivative for d_dx, d_dw only if label != 0 (marginalization step, or correct label)   
                        unsigned int off = c * itb + xtb;
                        const T* src_ptr = src + off;
                        T s = src_ptr[b];
                        T d_dy_val = expf(w + s) *exp_res * delta_ptr[b]; //res              // UND HIER
                    if ( (y < 0) || (c == y) ){
                        T* dst_ptr = dst + off;
                        if (d_dx) dst_ptr[b] = d_dy_val;
                        if (d_dw) temp_w_delta[threadIdx.x] += s_val * d_dy_val;
                    }
                    //except for d_dy, since it does not depend on the label
                    if (d_dy) Y_delta_ptr[c] += d_dy_val;                 

            }
            //logarithmic add of partial d_dw
            if (d_dw){
                    for ( unsigned int j = blockDim.x/2; j > 0; j/=2){
                        __syncthreads();
                                if (threadIdx.x < j){
                                    temp_w_delta[threadIdx.x] += temp_w_delta[threadIdx.x + j];
                                }
                    }
                    __syncthreads();
                        if (threadIdx.x == 0 ){
                                s += temp_w_delta[0];
                            }
                    //reset shared memory of this thead
                        temp_w_delta[threadIdx.x] = 0;
            }
        }

        //write result to global memory
        if (d_dw){
            if (threadIdx.x == 0) w_delta[c] = s;
        }       
}


template<class T>
void spn_output_op_grad_host(T* dst, const T* src, T* w_delta, T* Y_delta, const T* m_W, const T* Y, const T* S, const T* lae_res, const T* delta,
unsigned int lines, unsigned int items, unsigned int batch, const bool d_dx, const bool d_dy, const bool d_dw, float eps){
        unsigned int itb = (items * batch);
        int y;
        for ( unsigned int b = 0; b < batch; b ++){
            unsigned int btl = b * lines;
            const T* Y_ptr = Y + btl;
            y = int( Y_ptr[0] );
            T* Y_delta_ptr = Y_delta + b * btl;      
            T s_val = 1/(S[b] + eps);
            for ( unsigned int x = 0; x <items; x++){
                for ( unsigned int c = 0; c < lines; c++){
                    unsigned int xtb = x * batch;
                    unsigned int off = c* itb + xtb;
                    const T* src_ptr = src + off;
                    T* dst_ptr = dst + off;
                    const T* delta_ptr = delta + xtb;
                    const T* lae_ptr = lae_res + xtb;
                    T s = src_ptr[b];
                    T w = m_W[c];
                    T d_dy_val = ( expf(w+s)/expf(lae_ptr[b]) ) * delta_ptr[b];                    
                    if ( (y < 0) || (c == y)){
                        if (d_dx) dst_ptr[b] = d_dy_val;
                        if (d_dw) w_delta[c] += s_val * d_dy_val;
                    }
                    if (d_dy) Y_delta_ptr[c] +=  d_dy_val;

                }
            }
        }
}    
    
template<class V, class M, class T>
void spn_output_op_grad(tensor<V,M,T>& dst, const tensor<V,M,T>& src, tensor<V,M,T>& w_delta, tensor<V,M,T>& Y_delta, const tensor<V,M,T>& m_W, 
                        const tensor<V,M,T>& Y, const tensor<V,M,T>& S, const tensor<V,M,T>& lae_res, const tensor<V,M,T>& delta, const bool d_dx, const bool d_dw, const bool d_dy, float eps){
        
        cuvAssert((src.shape().size() == 3) || (src.shape().size() == 4));
        unsigned int lines = src.shape(0);
        unsigned int items;
        unsigned int batch;
        if (src.shape().size() == 3) { 
            items = src.shape(1);
            batch = src.shape(2);
        } else { 
            items = src.shape(1) * src.shape(2);
            batch = src.shape(3);
        }
 
        cuv::fill (dst, 0);
        cuv::fill (Y_delta, 0);
        cuv::fill (w_delta, 0);
        
        //check dimensions of dst
        cuvAssert(lines > 1);
        cuvAssert(items > 0);
        cuvAssert(batch > 0);   
        cuvAssert(src.shape(0) ==  Y.shape(1));
        cuvAssert((src.shape(2) == Y.shape(0)) || (src.shape(3) == Y.shape(0)) );
        cuvAssert(delta.shape(0) == 1);
        cuvAssert(delta.shape(1) ==  src.shape(1));
        cuvAssert(delta.shape(2) ==  src.shape(2));
        cuvAssert(m_W.shape(0) ==  src.shape(0));
        cuvAssert(Y.shape() ==  Y_delta.shape());
        cuvAssert(Y_delta.shape(1) > 1);
        cuvAssert(m_W.shape() ==  w_delta.shape());
        cuvAssert( (S.shape(0) ==  src.shape(2)) || (S.shape(0) ==  src.shape(3)));
        cuvAssert(src.shape() == dst.shape());
        
        if(IsSame<M,host_memory_space>::Result::value){
               spn_output_op_grad_host(dst.ptr(), src.ptr(), w_delta.ptr(), Y_delta.ptr(), m_W.ptr(), Y.ptr(), S.ptr(), lae_res.ptr(), delta.ptr(), lines, items, batch, d_dx, d_dy, d_dw, eps);
                
        }else{
            // device: run kernel
            unsigned int num_blocks  = lines;
            unsigned int num_threads = min(MAX_THREADS, int(32 * ceil( batch / 32. )));
            unsigned int sharedMemory  =  num_threads * sizeof(float);
            spn_output_op_grad_kernel<<<num_blocks,num_threads, sharedMemory>>>(dst.ptr(), src.ptr(), w_delta.ptr(), Y_delta.ptr(), m_W.ptr(), Y.ptr(), S.ptr(), lae_res.ptr(), delta.ptr(), lines, items, batch, d_dx, d_dy, d_dw, eps);

            cuvSafeCall(cudaThreadSynchronize());
        }
    }
    
    

// instantiate
#define  TENS(V,M,T)       tensor<V,M,T>
#define CTENS(V,M,T) const TENS(V,M,T)
#define INST(V,M,T) \
template void spn_output_op<V,M,T>(TENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&); \
template void spn_output_op_grad<V,M,T>(TENS(V,M,T)&, CTENS(V,M,T)&, TENS(V,M,T)&, TENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, bool, bool, bool, float); \
template void weighted_sub_tensor_op<V,M,T>(TENS(V,M,T)&, TENS(unsigned char,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, unsigned int, unsigned int, unsigned int, weighted_sub_tensor_op_functor, float); \
template void weighted_sub_tensor_op_grad<V,M,T>(TENS(V,M,T)&, TENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, CTENS(unsigned char,M,T)&, bool, bool,  bool, unsigned int, unsigned int, unsigned int, weighted_sub_tensor_op_functor, float); \
template void tuplewise_op<V,M,T>(TENS(V,M,T)&, CTENS(V,M,T)&, unsigned int, unsigned int, tuplewise_op_functor, float); \
template void tuplewise_op_grad<V,M,T>(TENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, unsigned int, unsigned int, tuplewise_op_functor, float); \
template void reorder_for_conv<V,M,T>(TENS(V,M,T)&, CTENS(V,M,T)&); \
template void reorder_from_conv<V,M,T>(TENS(V,M,T)&, CTENS(V,M,T)&); \
template void crop<V,M,T>(TENS(V,M,T)&, CTENS(V,M,T)&, int, int); \
template void resize_bilinear<V,M,T>(TENS(V,M,T)&, CTENS(V,M,T)&, float); \
template void project_to_ball<V,M,T>(TENS(V,M,T)&, float); \
template void contrast_normalization<V,M,T>(TENS(V,M,T)&, TENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, int, float, float); \
template void contrast_normalization_grad<V,M,T>(TENS(V,M,T)&, TENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, int, float, float, float, float); \
template void response_normalization<V,M,T>(TENS(V,M,T)&, TENS(V,M,T)&, CTENS(V,M,T)&, int, float, float); \
template void response_normalization_grad<V,M,T>(TENS(V,M,T)&, TENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, int, float, float, float, float); \
template void response_norm_cross_map<V,M,T>(TENS(V,M,T)&, TENS(V,M,T)&, CTENS(V,M,T)&, int, float, float, bool); \
template void response_norm_cross_map_grad<V,M,T>(TENS(V,M,T)&, TENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, int, float, float, bool, float, float); \
template void bed_of_nails<V,M,T>(TENS(V,M,T)&, CTENS(V,M,T)&, int, int, float, float); \
template void bed_of_nails_grad<V,M,T>(TENS(V,M,T)&, CTENS(V,M,T)&, int, int, float, float); \
template void gaussian_blur<V,M,T>(TENS(V,M,T)&, CTENS(V,M,T)&, CTENS(V,M,T)&, bool, float, float); \
template void convolve2d(TENS(V,M,T)& dst,CTENS(V,M,T)& img,CTENS(V,M,T)& filter, int paddingStart, unsigned int moduleStride, unsigned int nGroups, float factNew, float factOld); \
template void convolve2d(TENS(V,M,T)& dst,CTENS(V,M,T)& img,CTENS(V,M,T)& filter, CTENS(int,M,T)&, int paddingStart, unsigned int moduleStride, unsigned int nGroups, float factNew, float factOld); \
template void d_conv2d_dfilt(TENS(V,M,T)& dst_, CTENS(V,M,T)& delta, CTENS(V,M,T)&   input, CTENS(int,M,T)&, int paddingStart, unsigned int moduleStride, unsigned int nGroups, unsigned int partialSum, float factNew, float factOld);\
template void d_conv2d_dfilt(TENS(V,M,T)& dst_, CTENS(V,M,T)& delta, CTENS(V,M,T)&   input, int paddingStart, unsigned int moduleStride, unsigned int nGroups, unsigned int partialSum, float factNew, float factOld); \
template void d_conv2d_dimg(TENS(V,M,T)& dst, CTENS(V,M,T)&   delta, CTENS(V,M,T)&   filter, CTENS(int,M,T)&, int paddingStart, unsigned int moduleStride, unsigned int nGroups, float factNew,float factOld); \
template void d_conv2d_dimg(TENS(V,M,T)& dst, CTENS(V,M,T)&   delta, CTENS(V,M,T)&   filter, int paddingStart, unsigned int moduleStride, unsigned int nGroups, float factNew,float factOld);
INST(float,host_memory_space,row_major);
INST(float,dev_memory_space,row_major);
}}
