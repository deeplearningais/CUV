#ifndef __RBM__HPP__
#define __RBM__HPP__

#include<cuv/basics/tensor.hpp>

namespace cuv{
namespace libs{
	/// Restricted Boltzmann Machine (RBM)
namespace rbm{

	/**
	 * @addtogroup libs
	 * @{
	 * @addtogroup rbm
	 * @{
	 */

	/** 
	 * @namespace cuv::libs::rbm
	 * Utility functions for restricted Boltzmann machine
	 */

	/**
	 * set a matrix to consecutive binary numbers in the columns, starting with the number `start'
	 *
	 * @param m      the target matrix
	 * @param start  the value of the first column
	 */
	template <class __value_type, class __memory_space_type, class __memory_layout_type>
	void set_binary_sequence(tensor<__value_type,__memory_space_type,__memory_layout_type>& m, const int& start);

	/**
	 * apply sigmoid column-wise with the temperature specified for each column
	 *
	 * @param m    source and target matrix
	 * @param temp the temperature (one value per column)
	 */
	template <class __value_type, class __memory_space_type, class __memory_layout_type>
	void sigm_temperature(tensor<__value_type,__memory_space_type,__memory_layout_type>& m, const tensor<__value_type,__memory_space_type>& temp);

	/**
	 * simulate a local connectivity pattern.
	 *
	 * This simulates the local connectivity in a dense matrix by setting
	 * values to zero which are not part of the local connectivity.
	 *
	 * assume the lower layer image has dimension vx times vy and the upper
	 * layer has size hx times hy. The lower layer has vm maps, the upper layer
	 * has hm maps.
	 *
	 * @param m       the matrix, should have dimension vx*vy*vm times hx*hy*hm
	 * @param patchsize  size of local receptive field
	 * @param vx      as explained above
	 * @param vy      as explained above
	 * @param hx
	 * @param hy
	 * @param maxdist_from_main_dia reset everything further than this many maps away from central diagonal
	 * @param round
	 */
	template <class __value_type, class __memory_space_type, class __memory_layout_type>
	void set_local_connectivity_in_dense_matrix(tensor<__value_type,__memory_space_type,__memory_layout_type>& m, int patchsize, int vx, int vy, int hx, int hy, int maxdist_from_main_dia=1E6, bool round=false);



	/** 
	 * copy one matrix into another but only at specified positions.
	 *
	 * @param dst    the target matrix (N x M)
	 * @param src    the source matrix (N x M)
	 * @param rowidx contains row indices (M x B). The values at position b*offset+rowidx(m,b) will be copied.
	 * @param offset offset (see rowidx)
	 *
	 *
	 * This kernel can be used to compute serial gibbs updates of a laterally
	 * connected RBM hidden layer. Then, offset is the size of one map in the
	 * hidden layer, src is the actual result of the calculation and dst is the
	 * one where only selected values are changed.
	 *
	 * rowidx must be set to indices which are at most as large as the mapsize (<offset, that is).
	 * note that for at least _some_ consecutive read operations, rowidx is somewhat "transposed".
	 */
	template <class __value_type, class __memory_space_type, class __memory_layout_type>
	void copy_at_rowidx(tensor<__value_type,__memory_space_type,__memory_layout_type>& dst, const tensor<__value_type,__memory_space_type,__memory_layout_type>&  src, const tensor<typename tensor<__value_type,__memory_space_type,__memory_layout_type>::size_type,__memory_space_type,__memory_layout_type>& rowidx, const unsigned int offset);

	template <class __value_type, class __memory_space_type, class __memory_layout_type>
	void copy_redblack(tensor<__value_type,__memory_space_type,__memory_layout_type>& dst, const tensor<__value_type,__memory_space_type,__memory_layout_type>&  src, const unsigned int num_maps, const unsigned int color);

      /** 
       * @brief Bit-Flip a row of a column-major matrix
       * 
       * @param matrix Matrix to apply functor on
       * @param row	   row to flip
       * 
       * changes the matrix such that its m-th row is now (1-original mth row)
       *
       */
      template<class __value_type, class __memory_layout, class __memory_space_type>
              void bitflip(
              tensor<__value_type,__memory_layout,__memory_space_type> & matrix,
                              typename tensor<__value_type,__memory_layout,__memory_space_type>::size_type row);
      /**
       * @}
       * @}
       */
} } }


#endif /* __RBM__HPP__ */
