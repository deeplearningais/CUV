#ifndef __KMEANS__HPP__
#define __KMEANS__HPP__

namespace cuv{
namespace libs{
namespace kmeans{
	/** 
	 * @namespace cuv::libs::kmeans
	 * Utility functions for k-means clustering
	 */
	template<class __data_matrix_type, class __index_vector_type>
	void compute_clusters(__data_matrix_type& clusters, const __data_matrix_type& data, const __index_vector_type& indices);	
} } }
#endif /* __KMEANS__HPP__ */
