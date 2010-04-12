#ifndef MOVE_HPP_
#define MOVE_HPP_

namespace cuv
{

	/** 
	 * @defgroup imageops Operations on Images
	 * @brief Write a moved version of each image (a column in src) to dst.
	 *
	 * Assumptions: 
	 * - (n*num_maps by m) matrix, 
	 *   where n=image_width*image_height is an image
	 * - images are in RGBA interleaved format(num_maps=4), A channel is ignored.
	 * - images can also be in grayscale (num_maps=1).
	 *
	 * @todo previously non-existent pixels at the border are filled... how?
	 * 
	 * @}
	 */
	 
	 /** 
	 * @brief Shift images by given amount
	 * 
	 * @param dst where the moved images are written
	 * @param src unsigned char where original images are taken from
	 * @param src_image_size  width and height of image in source
	 * @param dst_image_size  width and height of image in destination
	 * @param num_maps  how many maps there are in src
	 * @param xshift how much to shift right
	 * @param yshift how much to shift down
	 */
	template<class __matrix_typeA, class __matrix_typeB>
	void image_move(__matrix_typeA& dst, const __matrix_typeB& src, 
			const unsigned int& src_image_size, 
			const unsigned int& dst_image_size, 
			const unsigned int& src_num_maps,
			const int& xshift, 
			const int& yshift);
};



#endif /* MOVE_HPP_ */

