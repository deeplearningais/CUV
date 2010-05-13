#include <basics/toeplitz_matrix.hpp>
#include <basics/dia_matrix.hpp>

namespace cuv{
	
	/**
	 * get the average of all diagonals of a diagonal matrix
	 * 
	 * @param dst the vector where the results are stored in
	 * @param dia the diagonal matrix where the diagonals are supposed to be summed
	 */
	template<class T, class M, class I>
	void avg_diagonals( cuv::vector<T,M,I>& dst, const cuv::dia_matrix<T,M>& dia );

	/**
	 * get the average of all diagonal matrix of a diagonal matrix
	 * 
	 * @param dst the toeplitz-matrix where the results are stored in
	 * @param dia the diagonal matrix where the diagonals are supposed to be summed
	 */
	template<class T, class M, class I>
	void avg_diagonals( cuv::toeplitz_matrix<T,M,I>& dst, const cuv::dia_matrix<T,M>& dia );
}
