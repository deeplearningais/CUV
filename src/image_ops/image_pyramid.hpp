#ifndef __IMAGE_PYRAMID_HPP__
#define __IMAGE_PYRAMID_HPP__

#include <basics/dense_matrix.hpp>
#include <basics/cuda_array.hpp>

namespace cuv{

template<class T,class S, class I>
void gaussian_pyramid_downsample(
	dense_matrix<T,row_major,S,I>& dst,
	const cuda_array<T,S,I>& src
);
template<class T,class S, class I>
void gaussian_pyramid_upsample(
	dense_matrix<T,row_major,S,I>& dst,
	const cuda_array<T,S,I>& src
);

}
#endif /* __IMAGE_PYRAMID_HPP__ */
