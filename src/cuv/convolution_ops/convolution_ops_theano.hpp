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


#ifndef __THEANO_CONVOLUTIONS_HPP__
#define __THEANO_CONVOLUTIONS_HPP__

#include <cuv.hpp>

namespace cuv{
/*
 * Wrappers for theano CUDA convolution functions
 */

/** @defgroup convolution_ops_theano Convolution and pooling operations
* @{
*/

namespace theano_conv{


/**
 * initializes cuda using theano implementation 
 *
 */
void initcuda();
/**
 * finalizes cuda using theano implementation 
 *
 */
void finalize_cuda();

/**
 * convolve a set of images with a set of filters
 *
 * @param out       (nImg, nFilt, nModulesY, nModulesX)
 * @param images       (nImg, nImgChan, nImgPixY, nImgPixX)
 * @param filter    (nFilt, nFiltChan, nFiltPixY, nFiltPixX)
 * @param mode  valid or full convolution
 * @param version  version of the convolution
 *
 */
void convolve_2d(cuv::tensor<float,cuv::dev_memory_space>& out, const cuv::tensor<float,cuv::dev_memory_space>& images, const cuv::tensor<float,cuv::dev_memory_space>& kern, const std::string& mode, int version=-1);

/**
 * determine the gradient of a convolution w.r.t. the inputs
 *
 *  @param dst (nImages, nImageColors, nImgPixY, nImgPixX)
 *  @param delta (nImg, nFilt, nModulesY, nModulesX)
 *  @param filters (nFilters, nFilterColors, filterPixelsY, filterPixelsX)
 *  @param mode  valid or full convolution
 */
void d_convolve_d_images(cuv::tensor<float,cuv::dev_memory_space>& images, const cuv::tensor<float,cuv::dev_memory_space>& out, const cuv::tensor<float,cuv::dev_memory_space>& kern, const std::string& mode);

/**
 * determine the gradient of a convolution w.r.t. the filters
 *
 *  @param kern  (nFilters, nInpMaps, filterPixelsY, filterPixelsX)
 *  @param images   (nImages, nInpMaps, nImgPixY, nImgPixX)
 *  @param out  (nImages, nFilters, nModulesY, nModulesX, )
 *  @param mode  valid or full convolution
 *
 */
void d_convolve_d_kern(cuv::tensor<float,cuv::dev_memory_space>& kern, const cuv::tensor<float,cuv::dev_memory_space>& images, const cuv::tensor<float,cuv::dev_memory_space>& out, const std::string& mode);

}
/** @} */ //end group convolution_ops_theano
}
#endif /* __THEANO_CONVOLUTIONS_HPP__ */
