#ifndef __SEPARABLE_CONVOLUTION_HPP__
#define __SEPARABLE_CONVOLUTION_HPP__

//#include <boost/ptr_container/ptr_vector.hpp>
#include <cuv/basics/tensor.hpp>
#include <cuv/basics/image.hpp>

namespace cuv{
	/// separable convolution
	namespace sep_conv
	{
		/**
		 * @addtogroup libs
		 * @{
		 * @addtogroup sep_conv
		 * @{
		 */

        /**
         * identifiers for different types of separable filters
         */
		enum separable_filter{
			SP_GAUSS,  ///< gaussian filter
			SP_CENTERED_DERIVATIVE, ///< centered derivative (-1,0,1)
			SP_BOX, ///< box filter
			SP_ORIENTATION_AND_MAGNITUDE, ///< orientation+magnitude (for HOG, primarily)
		};

        /**
         * separable convolution
         *
         * @param dst result matrix
         * @param src source (image)
         * @param radius filter radius (filter size is 2r+1 for radius r)
         * @param filt type of separable filter
         * @param axis the image dimension to apply this filter on
         * @param param optional filter parameter
         */
		template<class DstV, class SrcV, class M>
		void
		convolve(  tensor<DstV,M,row_major>& dst,
			   const tensor<SrcV,M,row_major>& src,
			   const unsigned int&   radius,
			   const separable_filter& filt, int axis=2,
			   const float& param=0.
		       	);

        /**
         * separable convolution
         *
         * @overload for interleaved images
         *
         * @param dst result image
         * @param src source image
         * @param radius filter radius (filter size is 2r+1 for radius r)
         * @param filt type of separable filter
         * @param axis the image dimension to apply this filter on
         * @param param optional filter parameter
         */
		template<int Channels, class DstV, class SrcV, class M>
		void
		convolve(        interleaved_image<Channels,DstV,M>& dst,
			   const interleaved_image<Channels,SrcV,M>& src,
			   const unsigned int&   radius,
			   const separable_filter& filt, int axis=2,
			   const float& param=0.
		       	);
		/**
		 * @}
		 * @}
		 */
	}
}

#endif /* __SEPARABLE_CONVOLUTION_HPP__ */



