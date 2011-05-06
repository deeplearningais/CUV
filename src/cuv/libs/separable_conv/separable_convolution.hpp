#ifndef __SEPARABLE_CONVOLUTION_HPP__
#define __SEPARABLE_CONVOLUTION_HPP__

//#include <boost/ptr_container/ptr_vector.hpp>
#include <cuv/basics/tensor.hpp>

namespace cuv{
	namespace sep_conv
	{
		enum separable_filter{
			SP_GAUSS,
			SP_CENTERED_DERIVATIVE,
			SP_BOX,
		};
		template<class DstV, class SrcV, class M, class A>
		void
		convolve(  tensor<DstV,M,row_major, A>& dst,
			   const tensor<SrcV,M,row_major, A>& src,
			   const unsigned int&   radius,
			   const separable_filter& filt, int axis=2,
			   const float& param=0.
		       	);
	}
}

#endif /* __SEPARABLE_CONVOLUTION_HPP__ */



