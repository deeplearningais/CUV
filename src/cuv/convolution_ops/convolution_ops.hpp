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





#ifndef __CONVOLUTION_OPS_HPP__
#define __CONVOLUTION_OPS_HPP__

#include <cuv/basics/tensor.hpp>
namespace cuv{

/*
 * Wrappers for Alex' CUDA convolution functions
 */

/** @defgroup convolution_ops Convolution and pooling operations
* @{
*/

/**
 * convolve a set of images with a set of filters
 *
 * @param dst       (nFilt, nModules, nImg)
 * @param img       (nImgChan, nImgPix, nImg)
 * @param filter    (nFiltChan, nFiltPix, nFilt)
 *
 */
template<class V, class M, class T>
    void
    convolve2d(tensor<V,M,T>& dst, const tensor<V,M,T>& img, const tensor<V,M,T>& filter, unsigned int paddingStart=0, unsigned int moduleStride=0, unsigned int nGroups=0);

/**
 * determine the gradient of a convolution w.r.t. the inputs
 *
 *  @param dst (nImageColors, imgPixels, nImages)
 *  @param delta (nFilt, nModules, nImg)
 *  @param filters (nFilterColors, filterPixels, nFilters)
 */
template<class V, class M, class T>
    void
    d_conv2d_dimg(tensor<V,M,T>& dst, const tensor<V,M,T>& delta, const tensor<V,M,T>& filter,
            unsigned int paddingStart=0, unsigned int moduleStride=0, unsigned int nGroups=0);

/**
 * determine the gradient of a convolution w.r.t. the filters
 *
 *  @param dst  (nModules/partialSum, nFilterColors, filterPixels, nFilters)
 *  @param input   (nImgColors, imgPixels, nImages), with stride given
 *  @param hidActs  (nFilters, numModules, nImages)
 *
 */
template<class V, class M, class T>
    void
    d_conv2d_dfilt(tensor<V,M,T>& dst, const tensor<V,M,T>& delta, const tensor<V,M,T>& input,
            unsigned int paddingStart=0,
            unsigned int moduleStride=0, unsigned int nGroups=0, unsigned int partialSum=1);

/**
 * two "simple" ways to do pooling in a network
 */
enum pool_type {
    PT_MAX, ///< local max-pooling
    PT_AVG  ///< local average pooling
};

/**
 * local pooling (average/max)
 *
 * @param images    (numFilters, imgPixels, numImages)
 * @param dst:      (numFilters, outputs, numImages)
 */
template<class V, class M, class T>
void local_pool(tensor<V,M,T>& dst, const tensor<V,M,T>& images,  
                   int subsX, int startX, int strideX, int outputsX, pool_type pooler);

/**
 * derivative of local max-pooling
 */
template<class V, class M, class T>
void local_max_pool_grad(tensor<V,M,T>& target, const tensor<V,M,T>& images, const tensor<V,M,T>& maxGrads,
        const tensor<V,M,T>& maxActs, int subsX, int startX, int strideX);

/**
 * derivative of local avg-pooling
 */
template<class V, class M, class T>
void local_avg_pool_grad(tensor<V,M,T>& target, const tensor<V,M,T>& avgGrads, 
        int subsX, int startX, int strideX);

/** @} */ //end group convolution_ops
}
#endif /* __CONVOLUTION_OPS_HPP__ */
