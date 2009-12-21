#ifndef __MATRIX_RPROP_HPP__
#define __MATRIX_RPROP_HPP__

#include <vector_ops/rprop.hpp>

namespace cuv{

template<class V, class O, class M, class I>
void rprop(dev_dense_matrix<V,M,I>& W,
		   dev_dense_matrix<V,M,I>& dW, 
		   dev_dense_matrix<O,M,I>& dW_old,
		   dev_dense_matrix<V,M,I>& rate){ rprop(W.vec(),dW.vec(),dW_old.vec(), rate.vec()); 
}

template<class V, class O, class M, class I>
void rprop(host_dense_matrix<V,M,I>&  W,
		   host_dense_matrix<V,M,I>& dW, 
		   host_dense_matrix<O,M,I>& dW_old,
		   host_dense_matrix<V,M,I>& rate){ rprop(W.vec(),dW.vec(),dW_old.vec(), rate.vec()); 
}
}

#endif /* __MATRIX_RPROP_HPP__ */
