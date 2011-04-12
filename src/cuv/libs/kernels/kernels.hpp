#ifndef __KERNELS__HPP__
#define __KERNELS__HPP__

#include <cuv/basics/tensor.hpp>

namespace cuv{
namespace libs{	
	namespace kernels{
	template <class __value_type, class __memory_space_type, class __memory_layout_type>
	void pairwise_distance(tensor<__value_type,__memory_space_type,__memory_layout_type>& result, const tensor<__value_type,__memory_space_type,__memory_layout_type>& A, const tensor<__value_type,__memory_space_type,__memory_layout_type>& B);
	}}}
#endif
