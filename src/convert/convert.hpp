#ifndef __CONVERT_HPP__
#define __CONVERT_HPP__

#include <cuv_general.hpp>
#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>

namespace cuv{

	template<class Dst, class Src>
	void convert(Dst& dst, const Src& src);
}

#endif /* __CONVERT_HPP__ */
