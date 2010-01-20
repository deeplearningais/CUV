#ifndef __CONVERT_HPP__
#define __CONVERT_HPP__

#include <tools/cuv_general.hpp>
#include <basics/dev_dense_matrix.hpp>
#include <basics/host_dense_matrix.hpp>

namespace cuv{

	/*
	 * Convert matrices or vectors
	 *  this looks a bit weird here, but what the hell.
	 *  Positive: it hides CUDA operations which have to be compiled by nvcc.
	 *  Negative: we have to instantiate _every_possible_use_ of this function in convert.cu.
	 */
	template<class Dst, class Src>
	void convert(Dst& dst, const Src& src);
}

#endif /* __CONVERT_HPP__ */
