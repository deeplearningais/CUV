#ifndef __SEPARABLE_CONVOLUTION_HPP__
#define __SEPARABLE_CONVOLUTION_HPP__

#include <boost/ptr_container/ptr_vector.hpp>
#include <cuv/basics/tensor.hpp>

namespace cuv{
	namespace sep_conv
	{
		enum separable_filter{
			SP_GAUSS,
			SP_SOBEL,
		};
		template<class DstV, class SrcV, class M>
		boost::ptr_vector<tensor<DstV,M,row_major> >
		convolve(  const tensor<SrcV,M,row_major>& src,
			   const unsigned int&   radius,
			   const separable_filter& filt );
	}
}

#endif /* __SEPARABLE_CONVOLUTION_HPP__ */



