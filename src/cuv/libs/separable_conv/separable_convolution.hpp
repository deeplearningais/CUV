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
		enum separable_filter{
			SP_GAUSS,
			SP_CENTERED_DERIVATIVE,
			SP_BOX,
			SP_ORIENTATION_AND_MAGNITUDE,
		};
		template<class DstV, class SrcV, class M, class A>
		void
		convolve(  tensor<DstV,M,row_major, A>& dst,
			   const tensor<SrcV,M,row_major, A>& src,
			   const unsigned int&   radius,
			   const separable_filter& filt, int axis=2,
			   const float& param=0.
		       	);

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



