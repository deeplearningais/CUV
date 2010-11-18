#ifndef __RBM__HPP__
#define __RBM__HPP__


namespace cuv{
namespace libs{
namespace rbm{
	/**
	 * set a matrix to consecutive binary numbers in the columns, starting with the number `start'
	 *
	 * @param m      the target matrix
	 * @param start  the value of the first column
	 */
	template<class __matrix_type>
	void set_binary_sequence(__matrix_type& m, const int& start);

	/**
	 * apply sigmoid column-wise with the temperature specified for each column
	 *
	 * @param m    source and target matrix
	 * @param temp the temperature (one value per column)
	 */
	template<class __matrix_type,class __vector_type>
	void sigm_temperature(__matrix_type& m, const __vector_type& temp);

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
	 * @param factor  equivalent to (hx*hv) / (vx*vy). So if the above image
	 *                size is half of the lower, factor is 0.25.
	 * @param vx      as explained above
	 * @param vy      as explained above
	 * @param maxdist_from_main_dia reset everything further than this many maps away from central diagonal
	 */
	template<class __matrix_type>
	void set_local_connectivity_in_dense_matrix(__matrix_type& m, int patchsize, int vx, int vy, int hx, int hy, int maxdist_from_main_dia=1E6, bool round=false);



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
	template<class __matrix_type,class __matrix_type2>
	void copy_at_rowidx(__matrix_type& dst, const __matrix_type&  src, const __matrix_type2& rowidx, const unsigned int offset);
} } }

#endif /* __RBM__HPP__ */
