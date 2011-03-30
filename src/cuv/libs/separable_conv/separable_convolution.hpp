#ifndef __SEPARABLE_CONVOLUTION_HPP__
#define __SEPARABLE_CONVOLUTION_HPP__

#include <boost/ptr_container/ptr_vector.hpp>
#include <cuv/basics/dense_matrix.hpp>

namespace cuv{
	namespace sep_conv
	{
		enum separable_filter{
			SP_GAUSS,
			SP_SOBEL,
		};
		template<class DstV, class SrcV, class M, class I>
		boost::ptr_vector<dense_matrix<DstV,row_major,M,I> >
		convolve(  const dense_matrix<SrcV,row_major,M,I>& src,
			   const unsigned int&   radius,
			   const separable_filter& filt );
	}
}

#endif /* __SEPARABLE_CONVOLUTION_HPP__ */



