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
 * wrappers of convolution operations by Alex Kriszevsky
 */
namespace alex_conv{

/**
 * Reorder memory for application of Alex' convolution routines.
 *
 * The routines by Alex require images to be in a slightly unintuitive memory order:
 * (nChannels, nPixH, nPixW, nImages). This is a convenience function to
 * change images of the form (nImages,nChannels,nPixH,nPixW) to the required
 * format at the cost of one transpose operation.
 *
 */
template<class V,class M, class T>
    void reorder_for_conv(tensor<V,M,T>& dst, const tensor<V,M,T>& src);

/**
 * Reverse operation of \c reorder_for_conv
 *
 */
template<class V,class M, class T>
    void reorder_from_conv(tensor<V,M,T>& dst, const tensor<V,M,T>& src);

/**
 * convolve a set of images with a set of filters
 *
 * @param dst       (nFilt, nModulesY, nModulesX, nImg)
 * @param img       (nImgChan, nImgPixY, nImgPixX, nImg)
 * @param filter    (nFiltChan, nFiltPix, nFilt)
 *
 */
template<class V, class M, class T>
    void
    convolve2d(tensor<V,M,T>& dst, const tensor<V,M,T>& img, const tensor<V,M,T>& filter, int paddingStart=0, unsigned int moduleStride=0, unsigned int nGroups=0, float factNew=1.f,float factOld=0.f);

/**
 * convolve a set of images with a set of filters, with arbitrary connectivity
 *
 * @param dst       (nFilt, nModulesY, nModulesX, nImg)
 * @param img       (nImgChan, nImgPixY, nImgPixX, nImg)
 * @param filter    (nFiltChan, nFiltPix, nFilt)
 * @param indices   (nGroups, nFiltChan) 
 *
 */
template<class V, class M, class T>
    void
    convolve2d(tensor<V,M,T>& dst, const tensor<V,M,T>& img, const tensor<V,M,T>& filter, 
            const tensor<int,M,T>& indices,
            int paddingStart=0, unsigned int moduleStride=0, unsigned int nGroups=0, float factNew=1.f,float factOld=0.f);

/**
 * determine the gradient of a convolution w.r.t. the inputs
 *
 *  @param dst (nImageColors, nImgPixY, nImgPixX, nImages)
 *  @param delta (nFilt, nModulesY, nModulesX, nImg)
 *  @param filters (nFilterColors, filterPixels, nFilters)
 */
template<class V, class M, class T>
    void
    d_conv2d_dimg(tensor<V,M,T>& dst, const tensor<V,M,T>& delta, const tensor<V,M,T>& filter,
            int paddingStart=0, unsigned int moduleStride=0, unsigned int nGroups=0, float factNew=1.f, float factOld=0.f);
/**
 * determine the gradient of a convolution wrt the inputs, with arbitrary connectivity
 *
 *  @param dst (nImageColors, nImgPixY, nImgPixX, nImages)
 *  @param delta (nFilt, nModulesY, nModulesX, nImg)
 *  @param filters (nFilterColors, filterPixels, nFilters)
 *  @param indices   (nGroups, nFiltChan) 
 */
template<class V, class M, class T>
    void
    d_conv2d_dimg(tensor<V,M,T>& dst, const tensor<V,M,T>& delta, const tensor<V,M,T>& filter,
            const tensor<int,M,T>& indices,
            int paddingStart=0, unsigned int moduleStride=0, unsigned int nGroups=0, float factNew=1.f, float factOld=0.f);

/**
 * determine the gradient of a convolution wrt the filters
 *
 *  @param dst  (nModules/partialSum, nFilterColors, filterPixels, nFilters)
 *  @param input   (nImgColors, nImgPixY, nImgPixX, nImages), with stride given
 *  @param hidActs  (nFilters, nModulesY, nModulesX, nImages)
 *
 */
template<class V, class M, class T>
    void
    d_conv2d_dfilt(tensor<V,M,T>& dst, const tensor<V,M,T>& delta, const tensor<V,M,T>& input,
            int paddingStart=0,
            unsigned int moduleStride=0, unsigned int nGroups=0, unsigned int partialSum=1, float factNew=1.f,float factOld=0.f);
/**
 * determine the gradient of a convolution wrt the filters, with arbitrary connectivity
 *
 *  @param dst  (nModules/partialSum, nFilterColors, filterPixels, nFilters)
 *  @param input   (nImgColors, nImgPixY, nImgPixX, nImages), with stride given
 *  @param hidActs  (nFilters, nModulesY, nModulesX, nImages)
 *  @param indices   (nGroups, nFiltChan) 
 *
 */
template<class V, class M, class T>
    void
    d_conv2d_dfilt(tensor<V,M,T>& dst, const tensor<V,M,T>& delta, const tensor<V,M,T>& input,
            const tensor<int,M,T>& indices,
            int paddingStart=0,
            unsigned int moduleStride=0, unsigned int nGroups=0, unsigned int partialSum=1, float factNew=1.f,float factOld=0.f);

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
 * @param images    (numFilters, nImgPixY, nImgPixX, numImages)
 * @param dst:      (numFilters, nImgPixY/n, nImgPixX/n, numImages)
 */
template<class V, class M, class T>
void local_pool(tensor<V,M,T>& dst, const tensor<V,M,T>& images,  
                   int subsX, int startX, int strideX, int outputsX, pool_type pooler);

/**
 * derivative of local max-pooling
 */
template<class V, class M, class T>
void local_max_pool_grad(tensor<V,M,T>& target, const tensor<V,M,T>& images, const tensor<V,M,T>& maxGrads,
        const tensor<V,M,T>& maxActs, int subsX, int startX, int strideX, float factNew=1.f, float factOld=0.f);

/**
 * derivative of local avg-pooling
 */
template<class V, class M, class T>
void local_avg_pool_grad(tensor<V,M,T>& target, const tensor<V,M,T>& avgGrads, 
        int subsX, int startX, int strideX);

/**
 * response normalization.
 *
 * in a local patch \f$\mathrm{Patch}(x)\f$ around \i x, calculates 
 * \f[ x' = \frac{x}{1 + \frac{\alpha}{|\mathrm{Patch}(x)|} \sum_{i\in\mathrm{Patch}(x)}  (x_i^2)^\beta}\f]
 *
 * @param target OUT \f$x'\f$
 * @param denoms OUT needed for gradient calculation, same shape as inputs
 * @param images IN inputs
 * @param patchSize IN width of (square) patches to operate on
 * @param float IN addScale \f$\alpha\f$
 * @param float IN powScale \f$\beta\f$
 */
template<class V, class M, class T>
void response_normalization(tensor<V,M,T>& target, tensor<V,M,T>& denoms, const tensor<V,M,T>& images, int patchSize, float addScale, float powScale);

/**
 * derivative of \c response_normalization.
 *
 * @param float OUT input_gradients the gradient w.r.t. \i x
 * @param float INOUT original_outputs (will be overwritten during calculation)
 * @param float IN original_inputs the original inputs to \c response_normalization
 * @param float IN denoms the intermediate result returned by \c response_normalization
 * @param float IN delta outer derivative of the current function (=backpropagated gradient)
 * @param float IN addScale \f$\alpha\f$
 * @param float IN powScale \f$\beta\f$
 */
template<class V, class M, class T>
void response_normalization_grad(tensor<V,M,T>& input_gradients, tensor<V,M,T>& original_outputs, const tensor<V,M,T>& original_inputs, 
        const tensor<V,M,T>& delta, const tensor<V,M,T>& denoms, int patchSize, float addScale, float powScale, float factNew=1.f, float factOld=0.f);

/**
 * contrast normalization.
 *
 * in a local patch \f$\mathrm{Patch}(x)\f$ around \i x, calculates 
 * \f[ x' = \frac{x}{1 + \frac{\alpha}{|\mathrm{Patch}(x)|} \sum_{i\in\mathrm{Patch}(x)}  (x_i^2)^\beta}\f]
 *
 * @param target OUT \f$x'\f$
 * @param denoms OUT needed for gradient calculation, same shape as inputs
 * @param meanDiffs IN difference to mean
 * @param images IN inputs
 * @param patchSize IN width of (square) patches to operate on
 * @param float IN addScale \f$\alpha\f$
 * @param float IN powScale \f$\beta\f$
 */
template<class V, class M, class T>
void contrast_normalization(tensor<V,M,T>& target, tensor<V,M,T>& denoms, const tensor<V,M,T>& meanDiffs, const tensor<V,M,T>& images, int patchSize, float addScale, float powScale);

/**
 * derivative of \c response_normalization.
 *
 * @param float OUT input_gradients the gradient w.r.t. \i x
 * @param float INOUT original_outputs (will be overwritten during calculation)
 * @param meanDiffs IN difference to mean
 * @param float IN original_inputs the original inputs to \c response_normalization
 * @param float IN denoms the intermediate result returned by \c response_normalization
 * @param float IN delta outer derivative of the current function (=backpropagated gradient)
 * @param float IN addScale \f$\alpha\f$
 * @param float IN powScale \f$\beta\f$
 */
template<class V, class M, class T>
void contrast_normalization_grad(tensor<V,M,T>& input_gradients, tensor<V,M,T>& original_outputs, const tensor<V,M,T>& meanDiffs,
        const tensor<V,M,T>& delta, const tensor<V,M,T>& denoms, int patchSize, float addScale, float powScale, float factNew=1.f, float factOld=0.f);

/**
 * response normalization accross maps.
 * @param target OUT normalized outputs are written here.
 * @param denoms OUT intermediate output used for gradient calculation
 * @param images IN the images which are to be normalized (4D: nChannels x nImgPixY x nImgPixX x nImg)
 * @param sizeF  IN the number of filters to normalize over
 */
template<class V, class M, class T>
void response_norm_cross_map(tensor<V,M,T>& target, tensor<V,M,T>& denoms, const tensor<V,M,T>& images, int sizeF, float addScale, float powScale, bool blocked);

/**
 * gradient of \c response_norm_cross_map
 */
template<class V, class M, class T>
void response_norm_cross_map_grad(tensor<V,M,T>& input_gradients, tensor<V,M,T>& original_outputs, const tensor<V,M,T>& original_inputs, 
        const tensor<V,M,T>& delta, const tensor<V,M,T>& denoms, int sizeF, float addScale, float powScale, bool blocked, float factNew=1.f, float factOld=0.f);

/**
 * gaussian blur (keeps size constant!).
 *
 * @param target OUT where blurred data is written to
 * @param images IN  (unblurred) inputs
 * @param filter IN  filter to convolve with (2k+1)
 * @param horiz  IN whether this is the horizontal or vertical filter pass
 * @param factNew IN  multiplier for newly calculated values
 * @param factOld IN  multiplier for data already in target
 */
template<class V, class M, class T>
void gaussian_blur(tensor<V,M,T>& target, const tensor<V,M,T>& images, const tensor<V,M,T>& filter, bool horiz, float factNew=1.f, float factOld=0.f);

/**
 * Bed of nails subsampling (take every n-th value in each direction).
 *
 * @param target OUT Where result is written to (smaller)
 * @param images IN  inputs (nChannels x nImgPixY x nImgPixX x nImg)
 * @param startX IN  where to start sampling
 * @param strideX IN  distance btw. picked values
 * @param factNew IN  multiplier for newly calculated values
 * @param factOld IN  multiplier for data already in target
 */
template<class V, class M, class T>
void bed_of_nails(tensor<V,M,T>& target, const tensor<V,M,T>& images, int startX, int strideX, float factNew=1.f, float factOld=0.f);

/**
 * Gradient of \c bed_of_nails
 *
 * @param target OUT Where result is written to (larger)
 * @param delta  IN outer derivative of current function
 * @param startX IN  where to start sampling
 * @param strideX IN  distance btw. picked values
 * @param factNew IN  multiplier for newly calculated values
 * @param factOld IN  multiplier for data already in target
 */
template<class V, class M, class T>
void bed_of_nails_grad(tensor<V,M,T>& target, const tensor<V,M,T>& delta, int startX, int strideX, float factNew=1.f, float factOld=0.f);

/**
 * cropping
 */
template<class V, class M, class T>
void crop(tensor<V,M,T>& cropped, const tensor<V,M,T>& images, int startY, int startX);

/**
 * project to ball.
 *
 * @param filters three-dimensional or four-dimensional filters kk
 *
 */
template<class V, class M, class T>
void project_to_ball(tensor<V,M,T>& filters, float ball);

/**
 * bilinear resizing
 */
template<class V, class M, class T>
void resize_bilinear(tensor<V,M,T>& dest, const tensor<V,M,T>& images, float scale);

/** 
 * @brief Determines which tuplewise operator to use.
 * @li TO_NORM calculates the norm 2  
 * @li TO_MAX calculates the maximum 
 * @li TO_ADD_SQUARED calculates the sum of the squares
 * @li TO_SUBSAMPLE subsamples the input over desired dimension
 *
 */
enum tuplewise_op_functor{
    TO_NORM,
    TO_MAX,
    TO_ADD_SQUARED,
    TO_SUBSAMPLE,
    TO_MEAN
};
/**
 * square the input, then add every map pair and take the square root.
 *
 * @param dst where to write result
 * @param src  the original input to the tuplewise_op
 * @param dim the dimension accross which we apply the tuplewise_op
 * @param subspace_size  the number of elements for which we calculate the norm
 * @param tuplewise_op_functor to the parameter determining wheater to calculate squared norm, norm or max out
 * @param eps small constant value which is added to the expression of the squared root
 */
template<class V, class M, class T>
void tuplewise_op(tensor<V,M,T>& dst, const tensor<V,M,T>& src, unsigned int dim, unsigned int subspace_size = 2, tuplewise_op_functor = TO_NORM, float eps = 0.f);

/**
 * calculates the gradient of tuplewise_op.
 *
 *
 * @param dst where to write result
 * @param X  the original input to the tuplewise_op
 * @param D  the backpropagated delta
 * @param dim the dimension accross which we apply the tuplewise_op
 * @param subspace_size  the number of elements for which we calculate the norm
 * @param tuplewise_op_functor to the parameter determining wheater to calculate squared norm, norm or max out
 * @param eps small constant value which is added to the expression of the squared root
 * 
 */
template<class V, class M, class T>
void tuplewise_op_grad(tensor<V,M,T>& dst, const tensor<V,M,T>& X, const tensor<V,M,T>& D, unsigned int dim, unsigned int subspace_size = 2, tuplewise_op_functor to = TO_NORM, float eps = 0.f);


/**
 * @brief Determines which weighted overlapping tuplewise operator to use.
 * @li WST_ADD calculates sum(wx)
 * @li WST_MAX calculates the log (max(wx)),  if (spn==true) the derivative calculates just the number of times w_i was max (needed for grad of SPN)
 * @li WST_MAX_LOGSPACE calculates the log (max(w + x), if (spn==true) the derivative calculates just the number of times w_i was max (needed for grad of SPN)
 * @li WST_LOGADDEXP calculates the  log (sum(exp(w * x))), if (spn==true) the derivative is weighted with 1/(S[1,y|X]+eps)
 * @li WST_LOGADDEXP_LOGSPACE calculates the log (sum(exp(w+x))) (thus it weights x with w in logspace), if (spn==true) the derivative is weighted with 1/(S[1,y|X]+eps)
 */
enum weighted_subtensor_functor{
    WST_WADD,
    WST_WMAX,
    WST_WMAX_LOGSPACE,
    WST_LOGWADDEXP,
    WST_LOGWADDEXP_LOGSPACE
};


/**
 * same as tuplewise op, but the input maps are weighted and the regions may overlap
 *
 * @param dst where to write result
 * @param dst_max_idx destintion to store max indices to
 * @param src  the original input to the weighted_subtensor_op
 * @param m_W weights (param 2) for weighted_subtensor_op
 * @param subspace_size  the number of elements for which we calculate the norm
 * @param tuplewise_op_functor to the parameter determining wheater to calculate squared norm, norm or max out
 * @param size, size of dst tensor
 * @param eps small constant value which is added to fprop for consistency
 */
template<class V, class M, class T>
void weighted_subtensor_op(tensor<V,M,T>& dst, tensor<unsigned char,M,T>& dst_max_idx, const tensor<V,M,T>& src, const tensor<V,M,T>& m_W, unsigned int size, unsigned int stride, unsigned int subspace_size = 2, weighted_subtensor_functor = WST_LOGWADDEXP, float eps = 0.00001f);

/**
 * calculates the gradient of weighted_subtensor_op.
 *
 *
 * @param dst where to write result
 * @param w_delta dst to write delta for weights (param[1])
 * @param X  the original input to the weighted_subtensor_op
 * @param D  the backpropagated delta
 * @param m_W the original weights given to weighted_subtensor_op
 * @param r0 result of weighted_subtensor_op fprop
 * @param S result of SPN ( only used in case bool spn=true
 * @param max_idx tensor where the max indices from fprop are stored
 * @param spn boolean which determines wheater normal op, or spn version is used
 * @param subspace_size  the number of elements for which we calculate the norm
 * @param tuplewise_op_functor to the parameter determining wheater to calculate squared norm, norm or max out
 * @param size, size of dst tensor* 
 * @param eps small constant value which is added to division 1/(sum exp(xw) + eps) for numerical stability
 * 
 */
template<class V, class M, class T>
void weighted_subtensor_op_grad(tensor<V,M,T>& dst, tensor<V,M,T>& w_delta, const tensor<V,M,T>& src, const tensor<V,M,T>& delta, const tensor<V,M,T>& m_W, const tensor<V,M,T>& r0, const tensor<V,M,T>& S, const tensor<unsigned char,M,T>& max_idx, const bool spn, const bool d_der, const bool w_der, unsigned int size, unsigned int stride, unsigned int subspace_size = 2, weighted_subtensor_functor to = WST_LOGWADDEXP, float eps = 0.00001f);

}

namespace misc_conv{
	template<class V, class M, class T>
	void upscaleOp(tensor<V,M,T>& dst, const tensor<V,M,T>& src, int factor);
	template<class V, class M, class T>
	void upscaleGrad(tensor<V,M,T>& dst, const tensor<V,M,T>& src, int factor);
}
/** @} */ //end group convolution_ops
}
#endif /* __CONVOLUTION_OPS_HPP__ */
