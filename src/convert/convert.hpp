#ifndef __CONVERT_HPP__
#define __CONVERT_HPP__

#include <tools/cuv_general.hpp>
#include <basics/dense_matrix.hpp>
namespace cuv{

	 /** @defgroup convert Convert matrices and vectors between different formats
 	 * @{
	 */
	 /** @brief Convert matrices and vectors between different formats
	 * 
	 * @param dst Destination
	 * @param src Source
	 *
	 * Converts between:
	 * 	-Column major an row major
	 * 	-Host and device matrices - which actually copies the memory to and from the GPU!
	 */
	template<class Dst, class Src>
	void convert(Dst& dst, const Src& src);
	 /** @} */ // end group convert
}

#endif /* __CONVERT_HPP__ */
