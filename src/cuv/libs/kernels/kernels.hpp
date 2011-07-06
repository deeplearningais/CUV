#ifndef __KERNELS__HPP__
#define __KERNELS__HPP__

#include <cuv/basics/tensor.hpp>

namespace cuv{
namespace libs{	
	namespace kernels{
	template <class __value_type, class __memory_space_type, class __memory_layout_type>
	void pairwise_distance_l2(tensor<__value_type,__memory_space_type,__memory_layout_type>& result, const tensor<__value_type,__memory_space_type,__memory_layout_type>& A, const tensor<__value_type,__memory_space_type,__memory_layout_type>& B);

	template <class __value_type, class __memory_space_type, class __memory_layout_type>
	void pairwise_distance_custom(tensor<__value_type,__memory_space_type,__memory_layout_type>& result, const tensor<__value_type,__memory_space_type,__memory_layout_type>& A, const tensor<__value_type,__memory_space_type,__memory_layout_type>& B);

	template <class __value_type, class __memory_space_type, class __memory_layout_type>
	tensor<__value_type,__memory_space_type,__memory_layout_type> pairwise_distance_l2(const tensor<__value_type,__memory_space_type,__memory_layout_type>& A, const tensor<__value_type,__memory_space_type,__memory_layout_type>& B){
             tensor<__value_type,__memory_space_type,__memory_layout_type>  result(extents[A.shape()[0]][B.shape()[0]]);
             pairwise_distance_l2(result, A, B);
             return result;
        }
	}}}
#endif
