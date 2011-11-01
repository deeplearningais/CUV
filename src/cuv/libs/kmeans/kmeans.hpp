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

	/**
	 * Compute the new cluster centers in one step.
	 * @param clusters the means to be updated
	 * @param data     the data of from which the means are computed
	 * @param indices  for every datapoint the index of the closest mean
	 */
	template<class __data_value_type, class __memory_space_type, class __memory_layout_type>
	void compute_clusters(cuv::tensor<__data_value_type, __memory_space_type, __memory_layout_type>& clusters,
		       	const cuv::tensor<__data_value_type, __memory_space_type, __memory_layout_type>& data,
		       	const cuv::tensor<typename cuv::tensor<__data_value_type, __memory_space_type, __memory_layout_type>::index_type,__memory_space_type>& indices);

	/**
	 * Sort the dataset according to the indices
	 * @param res      space for the sorted data
	 * @param data     the data which is to be sorted
	 * @param indices  for every datapoint the index of the closest mean
	 */
	template<class __data_value_type, class __memory_space_type, class __memory_layout_type>
	void sort_by_index(cuv::tensor<__data_value_type, __memory_space_type, __memory_layout_type>& res,
		       	cuv::tensor<typename cuv::tensor<__data_value_type, __memory_space_type, __memory_layout_type>::index_type,__memory_space_type>& indices,
		       	const cuv::tensor<__data_value_type, __memory_space_type, __memory_layout_type>& data);
} } }
#endif /* __KMEANS__HPP__ */
