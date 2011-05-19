#ifndef __KMEANS__HPP__
#define __KMEANS__HPP__

#include <cuv/basics/tensor.hpp>
namespace cuv{
namespace libs{
namespace kmeans{
	/** 
	 * @namespace cuv::libs::kmeans
	 * Utility functions for k-means clustering
	 */
	template<class __data_value_type, class __memory_space_type, class __memory_layout_type>
	void compute_clusters(cuv::tensor<__data_value_type, __memory_space_type, __memory_layout_type>& clusters,
		       	const cuv::tensor<__data_value_type, __memory_space_type, __memory_layout_type>& data,
		       	const cuv::tensor<typename cuv::tensor<__data_value_type, __memory_space_type, __memory_layout_type>::index_type,__memory_space_type>& indices);	
} } }
#endif /* __KMEANS__HPP__ */
