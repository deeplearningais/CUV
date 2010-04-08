#ifndef __CONVOLUTION_OPS_HPP__
#define __CONVOLUTION_OPS_HPP__

#include <stdio.h>

#include <vector_ops/vector_ops.hpp>
#include <basics/dense_matrix.hpp>
#include <basics/host_dense_matrix.hpp>

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
 * @param img		contains one input pattern in each row
 * @param filter	contains one filter in each row, number of filters must
 * 			        be multiples of 16.
 * @param dst		holds the target images of the convolution. one row for each
 *			        input image. width = dstSize^2 * numFilters
 */
template<class V, class M, class T, class I>
void convolve(dense_matrix<V,M,T,I>& dst,
		   dense_matrix<V,M,T,I>& img,
		   dense_matrix<V,M,T,I>& filter);

/**
  * @copydoc convolve(dense_matrix<V,M,T,I>& dst,dense_matrix<V,M,T,I>& img,dense_matrix<V,M,T,I>& filter)
  */
//template<class V, class M, class T, class I>
//void convolve(host_dense_matrix<V,M,T,I>& dst,
		   //host_dense_matrix<V,M,T,I>& img,
		   //host_dense_matrix<V,M,T,I>& filter);


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
template<class V, class M, class T, class I>
void convolve2(dense_matrix<V,M,T,I>& dst,
		   dense_matrix<V,M,T,I>& img,
		   dense_matrix<V,M,T,I>& filter,
		   int numFilters);

/**
  * @copydoc convolve2(dense_matrix<V,M,T,I>& dst, dense_matrix<V,M,T,I>& img, dense_matrix<V,M,T,I>& filter, int numFilters);
  */
//template<class V, class M, class T, class I>
//void convolve2(host_dense_matrix<V,M,T,I>& dst,
		   //host_dense_matrix<V,M,T,I>& img,
		   //host_dense_matrix<V,M,T,I>& filter,
		   //int numFilters);

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

template<class V, class M, class T, class I>
void convolve3(dense_matrix<V,M,T,I>& dst,
		   dense_matrix<V,M,T,I>& img,
		   dense_matrix<V,M,T,I>& filter);

/**
  * @copydoc convolve3(dense_matrix<V,M,T,I>& dst, dense_matrix<V,M,T,I>& img, dense_matrix<V,M,T,I>& filter);
  */
//template<class V, class M, class T, class I>
//void convolve3(host_dense_matrix<V,M,T,I>& dst,
		   //host_dense_matrix<V,M,T,I>& img,
		   //host_dense_matrix<V,M,T,I>& filter);
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
void prob_max_pooling(dev_vector<V,I>& sums, dense_matrix<V,M,T,I>& grid, int poolSize, bool sample);

/** 
 * @brief Reshape a matrix of images so that each column corresponds to a small window in the original image.
 * 
 * @param mat		Ouput matrix. Each column corresponds to a small window in grid. 
 * @param grid 		Input matrix. Each row corresponds to one image. 
 * @param poolSize	Size of window = mat.w()^2 
 *
 * Each image in grid is partitioned into grid.w() / poolSize^2 non-overlaping regions. These regions are saved in row major format into the columns of matrix.
 */
template<class V, class M, class T, class I>
void grid_to_matrix(dense_matrix<V,M,T,I>& mat,
		   dense_matrix<V,M,T,I>& grid,       
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
template<class V, class M, class T, class I>
void matrix_to_grid(dense_matrix<V,M,T,I>& grid,
		   dense_matrix<V,M,T,I>& mat,
		   int poolSize);

//template<class V, class M, class T, class I>
//void supersample(host_dense_matrix<V,M,T,I>& dst,
		//host_dense_matrix<V,M,T,I>& img,
		//int factor,
		//host_dense_matrix<int,row_major>* indices = NULL);

template<class V, class M, class T, class I>
void supersample(dense_matrix<V,M,T,I>& dst,
		dense_matrix<V,M,T,I>& img,
		int factor,
		dense_matrix<int,row_major,T>* indices = NULL);

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
template<class V, class M, class T, class I>
void reorder(dense_matrix<V,M,T,I>& A,
		   int blockLength);


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
//template<class V, class M, class T, class I>
//void reorder(host_dense_matrix<V,M,T,I>& A,
		   //int blockLength);


//template<class V, class M, class T, class I>
//void super_to_max(host_dense_matrix<V,M,T,I>& dst,
		//host_dense_matrix<V,M,T,I>& img,
		//int poolSize,
		//int overlap = 0,
		//host_dense_matrix<int,row_major>* indices = NULL,
		//host_dense_matrix<V,M,T,I>* filter = NULL);


template<class V, class M, class T, class I>
void super_to_max(dense_matrix<V,M,T,I>& dst,
		dense_matrix<V,M,T,I>& img,
		int poolSize,
		int overlap = 0,
		dense_matrix<int,row_major,T,I>* indices = NULL,
		dense_matrix<V,M,T,I>* filter = NULL);

template<class V, class M, class T, class I>
void copy_into(dense_matrix<V,M,T,I>& dst,
		   dense_matrix<V,M,T,I>& img,
		   int padding);

//template<class V, class M, class T, class I>
//void copy_into(host_dense_matrix<V,M,T,I>& dst,
		   //host_dense_matrix<V,M,T,I>& img,
		   //int padding);

//template<class V, class M, class T, class I>
//void max_pooling(host_dense_matrix<V,M,T,I>& dst,
		//host_dense_matrix<V,M,T,I>& img,
		//unsigned int poolSize,
		//unsigned int overlap = 0,
		//host_dense_matrix<int,row_major>* indices = NULL,
		//host_dense_matrix<V,M,T,I>* filter = NULL);

template<class V, class M, class T, class I>
void max_pooling(dense_matrix<V,M,T,I>& dst,
		dense_matrix<V,M,T,I>& img,
		unsigned int poolSize,
		unsigned int overlap = 0,
		dense_matrix<int,row_major,T,I>* indices = NULL,
		dense_matrix<V,M,T,I>* filter = NULL);


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
 * @brief Strips the padding inserted by copy_into
 * @param dst holds the stripped images. One row for each
 *            input image. width = dstSize^2 * numFilters
 * @param img contains one padded input pattern in each row
 * @param padding size of the padding
 *
 */
//template<class V, class M, class T, class I>
//void strip_padding(host_dense_matrix<V,M,T,I>& dst,
				   //host_dense_matrix<V,M,T,I>& img,
				   //unsigned int padding);


/**
 * @brief Fills a matrix with n copies of a given image
 * @param dst holds the target matrix with the n rows each containing the same source row
 * @param row is a vector containing the one row to be copied
 * @param n how often the row should be copied
 *
 */
template<class V, class M, class T,  class I>
void row_ncopy(dense_matrix<V,M,T,I>& dst,
			   dev_vector<V,I>& row,
			   unsigned int n);

/**
 * @brief Fills a matrix with n copies of a given image
 * @param dst holds the target matrix with the n rows each containing the same source row
 * @param row is a vector containing the one row to be copied
 * @param n how often the row should be copied
 *
 */
template<class V, class M, class T, class I>
void row_ncopy(dense_matrix<V,M,T,I>& dst,
			   host_vector<V,I>& row,
			   unsigned int n);

/**
 * @brief Inverts the filters in a filter matrix consisting of m filters in a row with n rows.
 * @param dst holds the target matrix with the inverted filters
 * @param filter is a matrix with m filters in a row and n rows
 * @param fs the filter size
 *
 */
template<class V, class M, class T, class I>
void filter_inverse(   dense_matrix<V,M,T,I>& dst,
					   dense_matrix<V,M,T,I>& filter,
					   unsigned int fs);

/**
 * @brief Inverts the filters in a filter matrix consisting of m filters in a row with n rows.
 * @param dst holds the target matrix with the inverted filters
 * @param filter is a matrix with m filters in a row and n rows
 * @param fs the filter size
 *
 */
//template<class V, class M, class T, class I>
//void filter_inverse(   host_dense_matrix<V,M,T,I>& dst,
					   //host_dense_matrix<V,M,T,I>& filter,
					   //unsigned int fs);

/**
 * @brief For a matrix with n maps in a row it returns a matrix where these maps are summed up into one map per row
 * @param dst holds the target matrix
 * @param mat is a matrix with n maps of the same size in a row
 * @param image_size the size of one map in that rows
 *
 */
template<class V, class M, class T, class I>
void add_maps_h(	dense_matrix<V,M,T,I>& dst,
					dense_matrix<V,M,T,I>& mat,
					unsigned int image_size);

/**
 * @brief For a matrix with n maps in a row it returns a matrix where these maps are summed up into one map per row
 * @param dst holds the target matrix
 * @param mat is a matrix with n maps of the same size in a row
 * @param image_size the size of one map in that rows
 *
 */
//template<class V, class M, class T, class I>
//void add_maps_h(	host_dense_matrix<V,M,T,I>& dst,
					//host_dense_matrix<V,M,T,I>& mat,
					//unsigned int image_size);
}

/** @} */ //end group convolution_ops
#endif /* __CONVOLUTION_OPS_HPP__ */
