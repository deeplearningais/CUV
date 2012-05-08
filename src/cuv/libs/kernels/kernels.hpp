#ifndef __KERNELS__HPP__
#define __KERNELS__HPP__

#include <cuv/basics/tensor.hpp>

namespace cuv{
namespace libs{	
	/// kernels
	namespace kernels{
	/**
	 * @addtogroup libs
	 * @{
	 * @addtogroup kernels
	 * @{
	 */
        /**
         * @brief determine pairwise distances between rows in argument matrices.
         *
         * @param result distance matrix (n_rows_A times n_rows_B)
         * @param A      first  matrix    (n_rows_A times K)
         * @param B      second matrix    (n_rows_B times K)
         * @param squared if true, do not determine square root of the distance
         */
	template <class __value_type, class __memory_space_type, class __memory_layout_type>
	void pairwise_distance_l2(tensor<__value_type,__memory_space_type,__memory_layout_type>& result, const tensor<__value_type,__memory_space_type,__memory_layout_type>& A, const tensor<__value_type,__memory_space_type,__memory_layout_type>& B, const bool & squared=false);

    /**
     * @brief more flexible implementation of @see pairwise_distance_l2.
     *
     * @param result distance matrix (n_rows_A times n_rows_B)
     * @param A      first  matrix    (n_rows_A times K)
     * @param B      second matrix    (n_rows_B times K)
     */
	template <class __value_type, class __memory_space_type, class __memory_layout_type>
	void pairwise_distance_custom(tensor<__value_type,__memory_space_type,__memory_layout_type>& result, const tensor<__value_type,__memory_space_type,__memory_layout_type>& A, const tensor<__value_type,__memory_space_type,__memory_layout_type>& B);

        /**
         * @brief determine pairwise distances between rows in argument matrices.
         *
         * @param A      first  matrix    (n_rows_A times K)
         * @param B      second matrix    (n_rows_B times K)
         * @param squared if true, do not determine square root of the distance
         *
         * @return distance matrix (n_rows_A times n_rows_B)
         */
	template <class __value_type, class __memory_space_type, class __memory_layout_type>
	tensor<__value_type,__memory_space_type,__memory_layout_type> pairwise_distance_l2(const tensor<__value_type,__memory_space_type,__memory_layout_type>& A, const tensor<__value_type,__memory_space_type,__memory_layout_type>& B, const bool & squared=false){
             tensor<__value_type,__memory_space_type,__memory_layout_type>  result(extents[A.shape(0)][B.shape(0)]);
             pairwise_distance_l2(result, A, B, squared);
             return result;
        }
	/**
	 * @}
	 * @}
	 */
	}}}
#endif
