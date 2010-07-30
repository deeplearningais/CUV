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

#include <stdio.h>

#include <vector_ops/vector_ops.hpp>
#include <basics/dense_matrix.hpp>

namespace cuv{

/*
 * Wrappers for Alex' CUDA convolution functions
 */

/** @defgroup convolution_ops Convolution and pooling operations
* @{
*/

/**
 * Convolve N patterns (images) with F filters, resulting in N*F target images.
 *
 * @param dst		holds the target images of the convolution. one row for each
 *			        filter, with one target image for each input image per row.
 * @param img		contains one input image in each row
 * @param filter	contains one filter in each row, number of filters must
 * 			        be multiples of 2. Routines is fastest for multiples of 16.
 * @param numGroups	amount of image groups. each group is convolved with it's own
 * 					set of filters.
 *
 *  The result is added to dst.
 */
template<class V, class M, class T, class I>
void convolve(dense_matrix<V,M,T,I>& dst,
		   dense_matrix<V,M,T,I>& img,
		   dense_matrix<V,M,T,I>& filter,
		   int numGroups = 1);


/** 
 * @brief Convolve N patterns (images), each with a unique set of F filters,
 *        resulting in N*F target images
 * @param dst holds the target images of the convolution. one row for each
 *            input image, with one target images for each filter per row.
 * @param img contains one input image in each row
 * @param filter contains N filters in each row, number of rows (filters) must
 *               be multiples of 2. Routines is fastest for multiples of 16.
 * @param numFilters number of filters (F)
 * @param numGroups	amount of image groups. each group is convolved with it's own
 * 					set of filters.
 *
 *  The result is added to dst.
 * 	This routine can be used to compute the weight gradients: img contains the
 *  activations from the lower layers, filters are the error maps from the upper
 *  layer. dst will then contain weight gradients for each pattern per row. (sum
 *  each column up).
 *
 */
template<class V, class M, class T, class I>
void convolve2(dense_matrix<V,M,T,I>& dst,
		   dense_matrix<V,M,T,I>& img,
		   dense_matrix<V,M,T,I>& filter,
		   int numFilters,
		   int numGroups = 1);


/** 
 * @brief Convolve N patterns, each consisting of F images (maps) with F filters
 * and add them up. Resulting in N target images
 *
 * @param dst 	holds the target images of the convolution. one row for each
 *            	input image, with one target image per row.
 * @param img 	contains N input images in each row
 * @param filter contains one filter in each row, number of filters must
 *               be multiples of 2. Routines is fastest for multiples of 16.
 * @param		numGroups	amount of image groups. each group is convolved with it's own
 * 				set of filters.
 *
 *  The result is added to dst.
 *  Filters are rotated by 180 degrees for the convolution. This routine is
 *  therefore useful to propagate errors back through convolutional layers:
 *  img contains the (padded) errors from the upper layers, filters are the
 *  convolutional filters. dst will then contain the backpropagated errors of
 *  the lower layer.
 */
template<class V, class M, class T, class I>
void convolve3(dense_matrix<V,M,T,I>& dst,
		   dense_matrix<V,M,T,I>& img,
		   dense_matrix<V,M,T,I>& filter,
		   int numGroups = 1);


/** 
 * @brief Sample from several multinomial distributions
 * 
 * @param grid Matrix of multinomial distributions.
 * 
 * Each row in grid corresponds to a random variable, each colum to a possible value.
 * The entries in each row have to be non-negative and sum to one.
 * The output matrix has one entry equal to one in each row and zeros everywhere else.
 * The probability of one entry in the output matrix being equal to one is exactly the value
 * of the corresponding entry in the input matrix.
 * 
 */
template<class V, class M, class T, class I>
void sample_multinomial(dense_matrix<V,M,T,I>& grid);


/** 
 * @brief Multinomial max-pooling as done by Lee (2009)
 * 
 * @param grid		 Input matrix where each column corresponds to one input image/filter response
 * @param poolSize	 size of the max-pooling window 
 * @param sample	 whether to sample from the multinomial distribution or just calculate the probabilities 
 * 
 * Each row in grid is interpreted as a square image and is partitioned in non-overlaping windows of size poolSize.
 * In each window the entries are normalized with a soft-max with an extra "hidden" pixel with value zero.
 * If sample is true, it is sampled from the resulting multinomial as described in sample_multinomial
 */
template<class V, class M, class T, class I>
void prob_max_pooling(dense_matrix<V,M,T,I>& grid, int poolSize, bool sample);

/** 
 * @brief Multinomial max-pooling as done by Lee (2009)
 *
 * @param sums		 Output matrix of max-pooled windows, width is the same as for grid, height is grid.h()/poolSize^2
 * @param grid		 Input matrix where each column corresponds to one input image/filter response
 * @param poolSize	 size of the max-pooling window 
 * @param sample	 whether to sample from the multinomial distribution or just calculate the probabilities 
 * 
 * Each row in grid is interpreted as a square image and is partitioned in non-overlaping windows of size poolSize.
 * In each window the entries are normalized with a soft-max with an extra "hidden" pixel with value zero.
 * If sample is true, it is sampled from the resulting multinomial as described in sample_multinomial.
 * Each window has a corresponding entry in sums. If sample is true, this entry is 1 iff any entry in the corresponding window is 1.
 * If sample is false the entry in sums is the sum of all entries in the corresponding window.
 */
template<class V, class M, class T, class I>
void prob_max_pooling(vector<V,T,I>& sums, dense_matrix<V,M,T,I>& grid, int poolSize, bool sample);


/** 
 * @brief Reshape a matrix of images so that each column corresponds to a small window in the original image.
 * 
 * @param mat		Ouput matrix. Each row corresponds to a small window in grid.
 * @param grid 		Input matrix. Each row corresponds to one image. 
 * @param poolSize	Size of window = mat.w()^2 
 *
 * Each image in grid is partitioned into grid.w() / poolSize^2 non-overlaping regions.
 * These regions are saved in row major format into the rows of a matrix.
 * Can be used to compute the sum over all pixels of each region.
 * Note: The entries in the output matrix are transposed, which shouldn't matter
 * for most purposes, e.g. reduction operations.
 */
template<class V, class M, class T, class I>
void grid_to_matrix(dense_matrix<V,M,T,I>& mat,
		   dense_matrix<V,M,T,I>& grid,       
		   int poolSize);


/** 
 * @brief Reshape a matrix of small windows in an image back to the original image.
 * 
 * @param grid		Output matrix, grid.h() = number of images, grid.w() = image size
 * @param mat		Input matrix, each row corresponds to one image region
 * @param poolSize	Size of window = mat.w()^2 
 * 
 * This is the inverse of grid_to_matrix. 
 */
template<class V, class M, class T, class I>
void matrix_to_grid(dense_matrix<V,M,T,I>& grid,
		   dense_matrix<V,M,T,I>& mat,
		   int poolSize);


/**
 * @brief Resize N images of size (m x m) by a factor s into images of size (m*s x m*s)
 *
 * @param dst		holds the output images. One row for each image of size (m*s x m*s)
 * @param img		contains the input images. One row for each image of size (m x m)
 * @param factor	Scaling factor
 * @param indices	matrix of indices. Each value corresponds to one block in the
 * 					supersampled image. Only the indexed pixel is filled with the
 * 					original value, any other pixel is set to zero.
 *
 * Supersampling takes a N x (m*m) matrix of N images of size (m x m) and enlarges
 * the images by a factor s. If no indices matrix is given, the input is simply
 * rescaled by the given factor.
 * With the index matrix, each pixel of the original image is only assigned to
 * one of the output pixel, depending on the index.
 */
template<class V, class M, class T, class I>
void supersample(dense_matrix<V,M,T,I>& dst,
		dense_matrix<V,M,T,I>& img,
		int factor,
		dense_matrix<int,row_major,T>* indices = NULL);


/**
 * @brief Resize N images of size (m x m) by a factor s into images of size (m/s x m/s)
 *
 * @param dst					holds the output images. One row for each image of size (m/s x m/s)
 * @param img					contains the input images. One row for each image of size (m x m)
 * @param factor				Scaling factor
 * @param avoidBankConflicts	The avoidBankConflicts option causes this function to use extra
 * 								shared memory to avoid all bank conflicts.
 *
 * Subsampling takes a N x (m*m) matrix of N images of size (m x m) and shrinks
 * the images by a factor s.
 * With the index matrix, each pixel of the original image is only assigned to
 * one of the output pixel, depending on the index.
 *
 * The avoidBankConflicts option causes this function to use extra shared memory to avoid all
 * bank conflicts. Most bank conflicts are avoided regardless of the setting of this parameter,
 * and so setting this parameter to true will have minimal impact on performance (Alex reported
 * a 5% improvement). (still can get 2-way conflicts if factor doesn't divide 16)
 *
 */
template<class V, class M, class T>
void subsample(dense_matrix<V,M,T>& dst,
		dense_matrix<V,M,T>& img,
		int factor,
		bool avoidBankConflicts);

/**
 * @brief Reorder blocks of data in a matrix
 * 
 * @param A target matrix 
 * @param blockLength size of each block
 *
 * sort the images in a matrix in a different order
 * input:  A1 B1 C1 D1
 *         A2 B2 C2 D2
 *         A3 B3 C3 D3
 * 		   where A1 is an image with blockLength pixels
 * output: A1
 *         A2
 *         A3
 *         B1
 *         B2
 *         ...
 */
template<class __matrix_type>
void reorder(__matrix_type& A,
		   int blockLength);

template<class __matrix_type>
void reorder(__matrix_type& dst,
		__matrix_type& src,
		int blockLength);



/**
 * @brief Propagate data from N smaller images into bigger images.
 *
 * @param dst		holds the output images. One row for each image
 * @param img		contains the input images. One row for each image of size (m x m)
 * @param poolSize	width of the overlapping square patches
 * @param overlap	amount of overlap
 * @param indices	matrix of indices. Each entry corresponds to one patch in the
 * 					supersampled image. Data from the input image is only propagated
 * 					to the designated index.
 * @param filter	Window function (matrix of poolSize x poolSize) to use. Data
 * 					in the target image is multiplied with this function.
 *
 * Each pixel from an input image is mapped to one pixel in the output image.
 * The exact position is determined by the index given in indices. If overlap > 0
 * more than one data element from img can be mapped to a pixel in dst, in which
 * case the values are added. Optionally, they are multiplied by a window fuction
 * given in filter.
 * This function can be used to perform the backpropagation step in a max pooling
 * layer.
  */
template<class V, class M, class T, class I>
void super_to_max(dense_matrix<V,M,T,I>& dst,
		dense_matrix<V,M,T,I>& img,
		int poolSize,
		int overlap = 0,
		dense_matrix<int,row_major,T,I>* indices = NULL,
		dense_matrix<V,M,T,I>* filter = NULL);



/**
 * @brief Copies images from img into dst and adds appropriate padding.
 *
 * @param dst		hold the output images, one in each row.
 * @param img		contains the input images, one in each row.
 * @param padding	Amount of pixels to be added on all sides
 *
 * Specifically, suppose "images" contains just one image and it looks like this:
 * IIII
 * IIII
 * IIII
 *
 * And targets looks like this:
 * XXXXXX
 * XXXXXX
 * XXXXXX
 * XXXXXX
 * XXXXXX
 *
 * After this function is called, targets will look like this:
 * XXXXXX
 * XIIIIX
 * XIIIIX
 * XIIIIX
 * XXXXXX
 *
 * Where the Is and Xs are arbitrary values.
 *
 * You can use this function to pad a bunch of images with a border of zeros. To do this,
 * the targets matrix should be all zeros.
 */
template<class V, class M, class T, class I>
void copy_into(dense_matrix<V,M,T,I>& dst,
		   dense_matrix<V,M,T,I>& img,
		   int padding);


/**
 * @brief Max pooling
 *
 * @param dst		holds the output images. One row for each image
 * @param img		contains the input images. One row for each image
 * @param poolSize	width of the overlapping square patches
 * @param overlap	amount of overlap
 * @param indices	matrix of indices. This matrix will store the index of it's
 * 					patche's maximum.
 * @param filter	Window function (matrix of poolSize x poolSize) to use.
 *
 * For each image in img, the maximum within each pooling window is calculated and
 * stored in dst. Pooling windows may be overlapping. If an index matrix is given,
 * the index of the maximum for each pool is stored in it and can later be used
 * for backpropagation using super_to_max. Optionally, a window function (filter) can
 * be applied prior to the maximum calculation.
 */
template<class V, class M, class T, class I>
void max_pooling(dense_matrix<V,M,T,I>& dst,
		dense_matrix<V,M,T,I>& img,
		unsigned int poolSize,
		unsigned int overlap = 0,
		dense_matrix<int,row_major,T,I>* indices = NULL,
		dense_matrix<V,M,T,I>* filter = NULL);

template<class V, class M, class T, class I>
void first_pooling(dense_matrix<V,M,T,I>& dst,
		dense_matrix<V,M,T,I>& img,
		unsigned int poolSize
		);

template<class V, class M, class T, class I>
void first_pooling_zeros( dense_matrix<V,M,T,I>& img,
		unsigned int poolSize
		);

/**
 * @brief Strips the padding inserted by copy_into
 * @param dst holds the stripped images. One row for each
 *            input image. width = dstSize^2 * numFilters
 * @param img contains one padded input pattern in each row
 * @param padding size of the padding
 *
 */
template<class V, class M, class T, class I>
void strip_padding(dense_matrix<V,M,T,I>& dst,
				   dense_matrix<V,M,T,I>& img,
				   unsigned int padding);


/**
 * @brief Fills a matrix with n copies of a given image
 * @param dst holds the target matrix with the n rows each containing the same source row
 * @param row is a vector containing the one row to be copied
 * @param n how often the row should be copied
 *
 */
template<class V, class M, class T,  class I>
void row_ncopy(dense_matrix<V,M,T,I>& dst,
			   vector<V,T,I>& row,
			   unsigned int n);

/**
 * @brief Fills the rows of the dst matrix with n copies of the row in the col matrix
 * @param dst holds the target matrix with the n*width rows each containing the same source row
 * @param col is a matrix containing the cols to be copied
 * @param n how often the row should be copied
 *
 */
template<class V, class M, class T,  class I>
void cols_ncopy(dense_matrix<V,M,T,I>& dst,
		dense_matrix<V,M,T,I>& col,
			   unsigned int n);



/**
 * @brief Rotates the filters in a filter matrix consisting of m filters in a row with n rows by 180 deg.
 * @param dst holds the target matrix with the inverted filters
 * @param filter is a matrix with n filters in a column and n columns
 * @param fs the filter size
 *
 */
template<class V, class M, class T, class I>
void filter_rotate(   dense_matrix<V,M,T,I>& dst,
					   dense_matrix<V,M,T,I>& filter,
					   unsigned int fs);


/**
 * @brief For a matrix with n maps in a row it returns a matrix where these maps are summed up into one map per row
 * @param dst holds the target matrix
 * @param mat is a matrix with n maps of the same size in a row
 * @param image_size the size of one map in that rows
 *
 */
//template<class V, class M, class T, class I>
//void add_maps_h(	dense_matrix<V,M,T,I>& dst,
//					dense_matrix<V,M,T,I>& mat,
//					unsigned int image_size);



/**
 * @brief calculates error maps
 * @param dst holds the target error matrices each in a row
 * @param img is a matrix with n maps to be compared with a blob
 * @param blob_mat a matrix holding the blob center information for each row in a row
 * @param interval_size multiplier of gaussian in: interval_size * (teacher) + interval_offset
 * @param interval_offset additive of gaussian in: interval_size * (teacher) + interval_offset
 *
 * For each map oder pic in img this function generates a map where the teacher is a gaussian with the corresponding coords in blob_mat as center
 */
template<class V, class M, class T, class I>
void calc_error_to_blob(				dense_matrix<V,M,T,I>& dst,
							dense_matrix<V,M,T,I>& img,
							dense_matrix<V,M,T,I>& blob_mat,
							unsigned int image_w,
							unsigned int image_h,
							unsigned int blob_size,
							float temporal_weight=1.0f,
							float interval_size=1.0f,
							float interval_offset=0.0f);

/**
 * @brief makes sure that the weights in the first numInhibitory filters are non-positive, the next numExitatory are non-negative
 * @param dst holds the target matrix for the corrected filters
 * @param filter is a matrix with n filters in a column and n columns
 * @param start_filter the filter col where to start checking (i.e. skipping input/output maps filters...)
 * @param num_inhibitory number of inhibitory filters
 * @param num_exitatory number of exitatory filters
 */

template<class V, class M, class T, class I>
void check_exitatory_inhibitory(
							dense_matrix<V,M,T,I>& filter,
							unsigned int start_filter,
							unsigned int filter_pixels,
							unsigned int num_inhibitory,
							unsigned int num_exitatory);


/**
 * @brief expects a random weight matrix and inverts positive entries in the first numInhibitory filters that are positive and in the next numExitatory those which are negative
 * @param dst holds the target matrix for the corrected filters
 * @param filter is a matrix with n filters in a column and n columns
 * @param start_filter the filter col where to start checking (i.e. skipping input/output maps filters...)
 * @param num_inhibitory number of inhibitory filters
 * @param num_exitatory number of exitatory filters
 */

template<class V, class M, class T, class I>
void init_exitatory_inhibitory(
							dense_matrix<V,M,T,I>& filter,
							unsigned int start_filter,
							unsigned int filter_pixels,
							unsigned int num_inhibitory,
							unsigned int num_exitatory);


}
/** @} */ //end group convolution_ops
#endif /* __CONVOLUTION_OPS_HPP__ */
