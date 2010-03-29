#ifndef __CONVOLUTION_OPS_HPP__
#define __CONVOLUTION_OPS_HPP__

#include <stdio.h>

#include <vector_ops/vector_ops.hpp>
#include <basics/dev_dense_matrix.hpp>
#include <basics/host_dense_matrix.hpp>

namespace cuv{

/*
 * Wrappers for Alex' CUDA convolution functions
 */

/**
 * Convolve N patterns (images) with F filters, resulting in N*F target images.
 *
 * @param img		contains one input pattern in each row
 * @param filters	contains one filter in each row, number of filters must
 * 			        be multiples of 16.
 * @param dst		holds the target images of the convolution. one row for each
 *			        input image. width = dstSize^2 * numFilters
 */
template<class V, class M, class I>
void convolve(dev_dense_matrix<V,M,I>& dst,
		   dev_dense_matrix<V,M,I>& img,
		   dev_dense_matrix<V,M,I>& filter);

/**
  * @copydoc convolve(dev_dense_matrix<float,row_major>&,dev_dense_matrix<float,row_major>&,dev_dense_matrix<float,row_major>&)
  */
template<class V, class M, class I>
void convolve(host_dense_matrix<V,M,I>& dst,
		   host_dense_matrix<V,M,I>& img,
		   host_dense_matrix<V,M,I>& filter);


/** 
 * @brief Convolve N patterns (images), each with a different set of numFilters filters,
 *        resulting in N*numFilter target images
 * @param dst holds the target images of the convolution. one row for each
 *            input image. width = dstSize^2 * numFilters
 * @param img contains one input pattern in each row
 * @param filter contains numFilter filters in each row, number of filters must
 *               be multiples of 16.
 * @param numFilters number of Filters 
 *
 * 	This routine can be used to compute the weight gradients: img contains the
 *  activations from the lower layers filters are the error maps from the upper
 *  layer. dst will then contain weight gradients for each pattern per row (sum
 *  each column up).
 *
 */
template<class V, class M, class I>
void convolve2(dev_dense_matrix<V,M,I>& dst,
		   dev_dense_matrix<V,M,I>& img,
		   dev_dense_matrix<V,M,I>& filter,
		   int numFilters);

/**
  * @copydoc convolve2(dev_dense_matrix<V,M,I>& dst, dev_dense_matrix<V,M,I>& img, dev_dense_matrix<V,M,I>& filter, int numFilters);
  */
template<class V, class M, class I>
void convolve2(host_dense_matrix<V,M,I>& dst,
		   host_dense_matrix<V,M,I>& img,
		   host_dense_matrix<V,M,I>& filter,
		   int numFilters);

/** 
 * @brief Convolve N patterns (images), each consisting of F images/maps with F filters
 * and add them up. Resulting in N target images
 *
 * @param dst 	holds the target images of the convolution. one row for each
 *            	input image. width = dstSize^2 
 * @param img 	contains F input pattern in each row
 * @param filter contains numFilter filters in each row, number of filters must
 *               be multiples of 16.
 *
 */

template<class V, class M, class I>
void convolve3(dev_dense_matrix<V,M,I>& dst,
		   dev_dense_matrix<V,M,I>& img,
		   dev_dense_matrix<V,M,I>& filter);

/**
  * @copydoc convolve3(dev_dense_matrix<V,M,I>& dst, dev_dense_matrix<V,M,I>& img, dev_dense_matrix<V,M,I>& filter);
  */
template<class V, class M, class I>
void convolve3(host_dense_matrix<V,M,I>& dst,
		   host_dense_matrix<V,M,I>& img,
		   host_dense_matrix<V,M,I>& filter);
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

template<class V, class M, class I>
void sample_multinomial(dev_dense_matrix<V,M,I>& grid);

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

template<class V, class M, class I>
void prob_max_pooling(dev_dense_matrix<V,M,I>& grid, int poolSize, bool sample);

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
template<class V, class M, class I>
void prob_max_pooling(dev_vector<V,I>& sums, dev_dense_matrix<V,M,I>& grid, int poolSize, bool sample);

/** 
 * @brief Reshape a matrix of images so that each column corresponds to a small window in the original image.
 * 
 * @param mat		Ouput matrix. Each column corresponds to a small window in grid. 
 * @param grid 		Input matrix. Each row corresponds to one image. 
 * @param poolSize	Size of window = mat.w()^2 
 *
 * Each image in grid is partitioned into grid.w() / poolSize^2 non-overlaping regions. These regions are saved in row major format into the columns of matrix.
 */
template<class V, class M, class I>
void grid_to_matrix(dev_dense_matrix<V,M,I>& mat,
		   dev_dense_matrix<V,M,I>& grid,       
		   int poolSize);
/** 
 * @brief Reshape a matrix of small windows in an image back to the original image.
 * 
 * @param grid		Output matrix, grid.w() = number of images 
 * @param mat		Input matrix 
 * @param poolSize	Size of window = mat.w()^2 
 * 
 * This is the inverse of grid_to_matrix. 
 */
template<class V, class M, class I>
void matrix_to_grid(dev_dense_matrix<V,M,I>& grid,
		   dev_dense_matrix<V,M,I>& mat,
		   int poolSize);

template<class V, class M, class I>
void supersample(host_dense_matrix<V,M,I>& dst,
		host_dense_matrix<V,M,I>& img,
		int factor,
		host_dense_matrix<int,row_major>* indices = NULL);

template<class V, class M, class I>
void supersample(dev_dense_matrix<V,M,I>& dst,
		dev_dense_matrix<V,M,I>& img,
		int factor,
		dev_dense_matrix<int,row_major>* indices = NULL);

/**
 * @brief Reorder blocks in a matrix
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
 *         ..
 */
template<class V, class M, class I>
void reorder(dev_dense_matrix<V,M,I>& A,
		   int blockLength);

/**
  * @copydoc reorder(dev_dense_matrix<V,M,I>& A, int blockLength);

  */
template<class V, class M, class I>
void reorder(host_dense_matrix<V,M,I>& A,
		   int blockLength);


template<class V, class M, class I>
void super_to_max(host_dense_matrix<V,M,I>& dst,
		host_dense_matrix<V,M,I>& img,
		int poolSize,
		int overlap = 0,
		host_dense_matrix<int,row_major>* indices = NULL,
		host_dense_matrix<V,M,I>* filter = NULL);


template<class V, class M, class I>
void super_to_max(dev_dense_matrix<V,M,I>& dst,
		dev_dense_matrix<V,M,I>& img,
		int poolSize,
		int overlap = 0,
		dev_dense_matrix<int,row_major>* indices = NULL,
		dev_dense_matrix<V,M,I>* filter = NULL);

template<class V, class M, class I>
void copy_into(dev_dense_matrix<V,M,I>& dst,
		   dev_dense_matrix<V,M,I>& img,
		   int padding);

template<class V, class M, class I>
void copy_into(host_dense_matrix<V,M,I>& dst,
		   host_dense_matrix<V,M,I>& img,
		   int padding);

template<class V, class M, class I>
void max_pooling(host_dense_matrix<V,M,I>& dst,
		host_dense_matrix<V,M,I>& img,
		unsigned int poolSize,
		unsigned int overlap = 0,
		host_dense_matrix<int,row_major>* indices = NULL,
		host_dense_matrix<V,M,I>* filter = NULL);

template<class V, class M, class I>
void max_pooling(dev_dense_matrix<V,M,I>& dst,
		dev_dense_matrix<V,M,I>& img,
		unsigned int poolSize,
		unsigned int overlap = 0,
		dev_dense_matrix<int,row_major>* indices = NULL,
		dev_dense_matrix<V,M,I>* filter = NULL);

template<class V, class M, class I>
void strip_padding(dev_dense_matrix<V,M,I>& dst,
				   dev_dense_matrix<V,M,I>& img,
				   unsigned int padding);

template<class V, class M, class I>
void strip_padding(host_dense_matrix<V,M,I>& dst,
				   host_dense_matrix<V,M,I>& img,
				   unsigned int padding);

template<class V, class M, class I>
void row_ncopy(dev_dense_matrix<V,M,I>& dst,
			   dev_vector<V,I>& row,
			   unsigned int n);

template<class V, class M, class I>
void row_ncopy(host_dense_matrix<V,M,I>& dst,
			   host_vector<V,I>& row,
			   unsigned int n);

template<class V, class M, class I>
void filter_inverse(   dev_dense_matrix<V,M,I>& dst,
					   dev_dense_matrix<V,M,I>& filter,
					   unsigned int fs);

template<class V, class M, class I>
void filter_inverse(   host_dense_matrix<V,M,I>& dst,
					   host_dense_matrix<V,M,I>& filter,
					   unsigned int fs);

template<class V, class M, class I>
void add_maps_h(	dev_dense_matrix<V,M,I>& dst,
					dev_dense_matrix<V,M,I>& mat,
					unsigned int image_size);

template<class V, class M, class I>
void add_maps_h(	host_dense_matrix<V,M,I>& dst,
					host_dense_matrix<V,M,I>& mat,
					unsigned int image_size);
}
#endif /* __CONVOLUTION_OPS_HPP__ */
