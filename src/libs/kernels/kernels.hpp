#ifndef __KERNELS__HPP__
#define __KERNELS__HPP__

namespace cuv{
namespace libs{	
	namespace kernels{
	template <class __matrix_type>
	void pairwise_distance(__matrix_type& result, const __matrix_type& A, const __matrix_type& B);
	}}}
#endif
