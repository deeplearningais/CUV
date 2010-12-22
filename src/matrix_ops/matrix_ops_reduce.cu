//*LB*
// Copyright (c) 2010, University of Bonn, Institute for Computer Science VI
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of the University of Bonn 
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*LE*

#include <stdio.h>
#include <stdexcept>

#include <cuv_general.hpp>
#include <meta_programming.hpp>
#include <vector_ops/functors.hpp>
#include "matrix_ops.hpp"

template<int BLOCK_SIZE, class T, class V, class RF>
__global__
void reduce_to_col_kernel(const T* matrix, V* vector, int nCols, int nRows,
		T param, T factNew, T factOld, RF reduce_functor) {

	__shared__ typename cuv::unconst<T>::type shared[BLOCK_SIZE / 2][BLOCK_SIZE * 2];

	V sum;

	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int ty = threadIdx.y;
	int by = blockIdx.y;

	const int row_idx = by * gridDim.x * blockDim.x +   	// offset according to y index in grid
						bx * blockDim.x +  					// offset according to block index
						tx;									// offset in block

	if (row_idx >= nRows)
		return;
	int off = blockDim.y;

	sum = cuv::reduce_functor_traits<T,RF>::init_value;
	for (int my = ty; my < nCols; my += off) {
		V f = matrix[my * nRows + row_idx ];
		sum=reduce_functor(sum,f);
	}

	shared[ty][tx] = sum;

	__syncthreads();

	int offset = blockDim.y / 2;
	while (offset > 0) {
		if (ty < offset) {
			shared[ty][tx]=reduce_functor(shared[ty][tx],shared[ty + offset][tx]);
		}
		offset >>= 1;
		__syncthreads();
	}
	
	if (ty == 0) {
		if (row_idx >= nCols){
		}
		if (cuv::reduce_functor_traits<T,RF>::returns_index)
			vector[row_idx] = shared[0][tx];
		else
			if(factOld != 0.f){
				vector[row_idx] = vector[row_idx] * factOld + shared[0][tx] * factNew;
			}else{
				vector[row_idx] = shared[0][tx] * factNew;
			}
	}
}

template<int BLOCK_SIZE, class T, class V, class RF>
__global__
void reduce_to_row_kernel(const T* matrix, V* vector, int nCols, int nRows,
		const T param, const T factNew, const T factOld, const RF reduce_functor) {
	__shared__ unsigned int indices[BLOCK_SIZE * BLOCK_SIZE];
	__shared__ typename cuv::unconst<T>::type values[BLOCK_SIZE * BLOCK_SIZE];
	const int tx = threadIdx.x, bx = blockIdx.x;
	const int ty = threadIdx.y, by = blockIdx.y;
	typedef typename cuv::functor_dispatcher<typename RF::functor_type> functor_dispatcher_type;
	functor_dispatcher_type func_disp;
	typedef typename cuv::reduce_functor_traits<T,RF> functor_traits;
	typedef typename functor_traits::result_type result_type;
	int off = blockDim.x;
	
	values[tx] = functor_traits::init_value;
	indices[tx] = 0;

	for (unsigned int my = tx; my < nRows; my += off) {
		T f = matrix[by * nRows + bx * blockDim.x + my];
			func_disp(reduce_functor,values[tx],indices[tx],f,my);
	}
	__syncthreads();

	int offset = blockDim.x / 2;
	while (offset > 0) {
		if (tx < offset) {
			func_disp(reduce_functor,values[tx],indices[tx],values[tx+offset],indices[tx+offset]);
		}
		offset >>= 1;
		__syncthreads();
	}
	if (tx == 0) {
		if (functor_traits::returns_index)
			vector[by * blockDim.y + ty] = indices[0];
		else{
			if(factOld != 0){
				vector[by * blockDim.y + ty] = vector[by * blockDim.y + ty]
					* factOld + values[0] * factNew;
			}else{
				vector[by * blockDim.y + ty] = values[0] * factNew;
			}
		}
	}
	__syncthreads();
}

template<unsigned int BLOCK_DIM, class I, class T>
__global__
void argmax_row_kernel(I* vector, const T* matrix, unsigned int nCols, unsigned int nRows) {
	__shared__ I shIdx[BLOCK_DIM]; // index of the maximum
	__shared__ T shVal[BLOCK_DIM]; // value

	const unsigned int tx = threadIdx.x;
	const unsigned int by = blockIdx.x + gridDim.x*blockIdx.y;
	if (by >= nCols)
	   return;
	const unsigned int off = blockDim.x;

	unsigned int idx = by * nRows + tx;
	shVal[tx] = (tx<nRows) ? matrix[idx] : (T) INT_MIN;
	shIdx[tx] = (tx<nRows) ? tx          : 0;

	for (unsigned int my = tx + off; my < nRows; my += off) {
	   idx += off;
	   T f = matrix[idx];

	   if (f > shVal[tx]) {
		  shVal[tx] = f;
		  shIdx[tx] = my;
	   }
	}
	__syncthreads();

	for (unsigned int offset = BLOCK_DIM/2 ; offset > 0; offset/=2) {
	   if (tx < offset) {
		   const unsigned int v = tx+offset;
		   if (shVal[tx] < shVal[v]) {
			   shVal[tx] = shVal[v];
			   shIdx[tx] = shIdx[v];
		   }
	   }
	}
	__syncthreads();

	if (tx == 0)
	   vector[by] = shIdx[0];
}

namespace cuv {

namespace reduce_to_col_impl {

	template<class V,class I, class V2, class RF>
	void reduce_to_col(vector<V2,host_memory_space,I>&v, const dense_matrix<V,column_major,host_memory_space,I>& m, const V& factNew, const V& factOld, RF reduce_functor) {
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.h() == v.size());
		typedef typename unconst<V>::type unconstV;
		vector<unconstV,host_memory_space,I> indices(v.size());
		typedef typename cuv::functor_dispatcher<typename RF::functor_type> functor_dispatcher_type;
		functor_dispatcher_type func_disp;
		typedef typename cuv::reduce_functor_traits<V,RF> functor_traits;
		const V* A_ptr = m.ptr();
		vector<V2,host_memory_space,I> old(v); // copy old vector for factOld and factNew computations
		V2* values_ptr = v.ptr();
		unconstV* indices_ptr = indices.ptr();

		for(int j=0; j<v.size(); j++) 
			*values_ptr++ =reduce_functor_traits<V,RF>::init_value; // initialize column vector

		for(int i=0;i<m.w();i++) {
			values_ptr = v.ptr();
			for(int j=0; j<m.h(); j++,A_ptr++,values_ptr++) 
				func_disp(reduce_functor,*values_ptr,*indices_ptr,*A_ptr,j);
		}

		values_ptr = v.ptr();
		V2* old_ptr = old.ptr();
		indices_ptr = indices.ptr();

		if (!reduce_functor_traits<V,RF>::returns_index) 
			if (factOld!=0)
				for(int j=0; j<v.size(); j++,values_ptr++,old_ptr++) {
					*values_ptr = factOld * *old_ptr + factNew * *values_ptr;
				}
			else
				for(int j=0; j<v.size(); j++,values_ptr++,old_ptr++) {
					*values_ptr = factNew * *values_ptr;
				}
		else
				for(int j=0; j<v.size(); j++,values_ptr++,indices_ptr++) {
					*values_ptr = *indices_ptr;
				}

	}
	template<class V,class I, class V2, class RF>
	void reduce_to_col(vector<V2,dev_memory_space,I>&v, const dense_matrix<V,column_major,dev_memory_space,I>& m, const V& factNew, const V& factOld, RF reduce_functor) {
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.h() == v.size());
		static const int BLOCK_SIZE = 16;
		static const int BLOCK_DIM_X = BLOCK_SIZE*2;
		static const int BLOCK_DIM_Y = BLOCK_SIZE/2;
		const int blocks_needed = ceil((float)m.h()/(BLOCK_DIM_X));
		int grid_x =0, grid_y=0;

		// how to handle grid dimension constraint
		if (blocks_needed <= 65535){
			grid_x = blocks_needed;
			grid_y = 1;
		}else{
			// try to avoid large noop blocks by adjusting x and y dimension to nearly equal size
			grid_x = ceil(sqrt(blocks_needed));
			grid_y = ceil((float)blocks_needed/grid_x);
		}
		dim3 grid(grid_x, grid_y);
		dim3 threads(BLOCK_DIM_X,BLOCK_DIM_Y);
		reduce_to_col_kernel<BLOCK_SIZE,V><<<grid,threads>>>(m.ptr(),v.ptr(),m.w(),m.h(),0,factNew,factOld,reduce_functor);
		cuvSafeCall(cudaThreadSynchronize());
	}
	// Dummy functions so that everything compiles:
	// actually only column major functions are ever used. row major cases are maped to "transposed" colum major case.
	template<class V,class I, class V2, class RF>
	void reduce_to_col(vector<V2,dev_memory_space,I>&v, const dense_matrix<V,row_major,dev_memory_space,I>& m, const V& factNew, const V& factOld, RF reduce_functor) {
		cuvAssert(false);
	}
	template<class V,class I, class V2, class RF>
	void reduce_to_col(vector<V2,host_memory_space,I>&v, const dense_matrix<V,row_major,host_memory_space,I>& m, const V& factNew, const V& factOld, RF reduce_functor) {
		cuvAssert(false);
	}

}//namespace reduce_to_col_imp

namespace reduce_to_row_impl {
	template<class V,class I, class V2, class RF>
	void reduce_to_row(vector<V2,host_memory_space,I>&v, const dense_matrix<V,column_major,host_memory_space,I>& m, const V& factNew, const V& factOld, RF reduce_functor) {
		cuvAssert(v.size()==m.w());
		typedef typename unconst<V>::type unconstV;
		vector<unconstV,host_memory_space,I> indices(v.size());
		typedef typename cuv::functor_dispatcher<typename RF::functor_type> functor_dispatcher_type;
		typedef typename cuv::reduce_functor_traits<V,RF> functor_traits;
		const V* A_ptr = m.ptr();
		vector<V2,host_memory_space,I> old(v); // copy old vector for factOld and factNew computations
		V2* values_ptr = v.ptr();
		unconstV* indices_ptr = indices.ptr();
		V2* old_ptr = old.ptr();

		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.w() == v.size());

		for(int j=0; j<v.size(); j++) 
			*values_ptr++ =reduce_functor_traits<V,RF>::init_value; // initialize column vector

		values_ptr = v.ptr();

		for(int i=0;i<m.w();i++, values_ptr++) {
			for(int j=0; j<m.h(); j++, A_ptr++){
					*values_ptr = reduce_functor(*A_ptr,*values_ptr);
			}
		}
		values_ptr = v.ptr();
		indices_ptr = indices.ptr();

		if (!reduce_functor_traits<V,RF>::returns_index){ 
			if (factOld!=0)
				for(int j=0; j<v.size(); j++,values_ptr++,old_ptr++) {
					*values_ptr = factOld * *old_ptr + factNew * *values_ptr;
				}
			else
				for(int j=0; j<v.size(); j++,values_ptr++) {
					*values_ptr = factNew * *values_ptr;
				}
		}
		else
			for(int j=0; j<v.size(); j++,values_ptr++,indices_ptr++)
				*values_ptr = *indices_ptr;

	}

	template<class V,class I, class V2, class RF>
	void reduce_to_row(vector<V2,dev_memory_space,I>&v, const dense_matrix<V,column_major,dev_memory_space,I>& m, const V& factNew, const V& factOld, RF reduce_functor) {
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.w() == v.size());
		static const int BLOCK_SIZE = 16;
		dim3 grid(1, m.w());
		dim3 threads(BLOCK_SIZE*BLOCK_SIZE,1);
		reduce_to_row_kernel<BLOCK_SIZE,V><<<grid,threads>>>(m.ptr(),v.ptr(),m.w(),m.h(),0,factNew,factOld,reduce_functor);
		cuvSafeCall(cudaThreadSynchronize());
	}

	// Dummy functions so that everything compiles:
	// actually only column major functions are ever used. row major cases are maped to "transposed" colum major case.
	template<class V,class I, class V2, class RF>
	void reduce_to_row(vector<V2,dev_memory_space,I>&v, const dense_matrix<V,row_major,dev_memory_space,I>& m, const V& factNew, const V& factOld, RF reduce_functor) {
		cuvAssert(false);
	}
	template<class V,class I, class V2, class RF>
	void reduce_to_row(vector<V2,host_memory_space,I>&v, const dense_matrix<V,row_major,host_memory_space,I>& m, const V& factNew, const V& factOld, RF reduce_functor) {
		cuvAssert(false);
	}

}//namespace reduce_to_row_imp

template<class __matrix_type, class __vector_type>
void reduce_to_col(__vector_type&v, const __matrix_type& m, reduce_functor rf, const typename __matrix_type::value_type& factNew, const typename __matrix_type::value_type& factOld) {
	typedef typename __matrix_type::value_type mat_val;
	typedef typename __matrix_type::index_type mat_ind;
	typedef typename __matrix_type::memory_layout mat_mem;
	typedef typename __vector_type::value_type vec_val;
	typedef typename __vector_type::index_type vec_ind;
	if (IsSame<mat_mem,row_major>::Result::value){
		//matrix is row major
		//create column major view and call reduce_to_row for column major
		const dense_matrix<const mat_val,column_major,typename __matrix_type::memory_space_type,mat_ind> cm_view(m.w(),m.h(),m.ptr(),true);
		reduce_to_row(v,cm_view,rf,factNew,factOld);
	}
	else{
		//matrix is column major
		switch(rf) {
			case RF_ADD:
			reduce_to_col_impl::reduce_to_col<mat_val,mat_ind,vec_val>(v,m,factNew,factOld,bf_plus<typename __matrix_type::value_type,typename __matrix_type::value_type>());
			break;
			case RF_ADD_SQUARED:
			reduce_to_col_impl::reduce_to_col<mat_val,mat_ind,vec_val>(v,m,factNew,factOld,bf_add_square<typename __matrix_type::value_type,typename __matrix_type::value_type>());
			break;
			case RF_MIN:
			reduce_to_col_impl::reduce_to_col<mat_val,mat_ind,vec_val>(v,m,factNew,factOld,bf_min<typename __matrix_type::value_type,typename __matrix_type::value_type>());
			break;
			case RF_MAX:
			reduce_to_col_impl::reduce_to_col<mat_val,mat_ind,vec_val>(v,m,factNew,factOld,bf_max<typename __matrix_type::value_type,typename __matrix_type::value_type>());
			break;
			default:
			throw std::runtime_error("supplied reduce_functor does not exist");
		}
	}
}

template<class __matrix_type, class __vector_type>
void reduce_to_row(__vector_type&v, const __matrix_type& m,reduce_functor rf, const typename __matrix_type::value_type& factNew, const typename __matrix_type::value_type& factOld) {
	
	typedef typename __matrix_type::value_type mat_val;
	typedef typename __matrix_type::index_type mat_ind;
	typedef typename __matrix_type::memory_layout mat_mem;
	typedef typename __vector_type::value_type vec_val;
	typedef typename __vector_type::index_type vec_ind;
	if (IsSame<mat_mem,row_major>::Result::value){
		//matrix is row major
		//create column major view and call reduce_to_col for column major
		const dense_matrix<const mat_val,column_major,typename __matrix_type::memory_space_type,mat_ind> cm_view(m.w(),m.h(),m.ptr(),true);
		reduce_to_col(v,cm_view,rf,factNew,factOld);
		}
	else{
		//matrix is column major
		switch(rf) {
			case RF_ADD:
			reduce_to_row_impl::reduce_to_row<mat_val,mat_ind,vec_val>(v,m,factNew,factOld,bf_plus<mat_val,mat_val>());
			break;
			case RF_ADD_SQUARED:
			reduce_to_row_impl::reduce_to_row<mat_val,mat_ind,vec_val>(v,m,factNew,factOld,bf_add_square<mat_val,mat_val>());
			break;
			case RF_MIN:
			reduce_to_row_impl::reduce_to_row<mat_val,mat_ind,vec_val>(v,m,factNew,factOld,bf_min<typename __matrix_type::value_type,typename __matrix_type::value_type>());
			break;
			case RF_MAX:
			reduce_to_row_impl::reduce_to_row<mat_val,mat_ind,vec_val>(v,m,factNew,factOld,bf_max<typename __matrix_type::value_type,typename __matrix_type::value_type>());
			break;
			//case RF_ARGMAX:
			//reduce_to_row_impl::reduce_to_row(v,m,factNew,factOld,reduce_argmax<vec_val,mat_val,mat_ind>());
			//break;
			//case RF_ARGMIN:
			//reduce_to_row_impl::reduce_to_row(v,m,factNew,factOld,reduce_argmin<vec_val,mat_val,mat_ind>());
			//break;
			default:
			throw std::runtime_error("supplied reduce_functor does not exist");
		}
	}
}

namespace argmax_to_XXX_impl{

	template<class V, class V2, class I>
	void argmax_to_row(vector<V2,dev_memory_space>&v, const dense_matrix<V,column_major, dev_memory_space, I>& m) {
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.w() == v.size());
		const unsigned int u = min(m.w(), MAX_GRID_SIZE);
		dim3 grid(u, ceil(m.w()/(float)u));
		static const unsigned int BLOCK_DIM = 256;
		argmax_row_kernel<BLOCK_DIM><<<grid,BLOCK_DIM>>>(v.ptr(),m.ptr(),m.w(),m.h());
		cuvSafeCall(cudaThreadSynchronize());
	}

	template<class V, class V2, class I>
	void argmax_to_column(vector<V2,dev_memory_space, I>&v, const dense_matrix<V,row_major,dev_memory_space,I>& m) {
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.h() == v.size());
		const unsigned int u = min(m.h(), MAX_GRID_SIZE);
		dim3 grid(u, ceil(m.h()/(float)u));
		static const unsigned int BLOCK_DIM = 256;
		argmax_row_kernel<BLOCK_DIM><<<grid,BLOCK_DIM>>>(v.ptr(),m.ptr(),m.h(),m.w());
		cuvSafeCall(cudaThreadSynchronize());
	}

	template<class V, class V2, class I>
	void argmax_to_row(vector<V2,host_memory_space,I>&v, const dense_matrix<V,column_major, host_memory_space,I>& m) {
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.w() == v.size());
		const V* ptr = m.ptr();
		V2* res = v.ptr();
		for(int i=0; i<m.w();i++) {
			int idx = 0;
			V val = *ptr;
			for(int j=0; j<m.h();j++) {
				if(*ptr > val) {
					val = *ptr;
					idx = j;
				}
				ptr++;
			}
			*res++ = idx;
		}
	}

	template<class V, class V2, class I>
	void argmax_to_column(vector<V2,host_memory_space,I>&v, const dense_matrix<V,row_major,host_memory_space,I>& m) {
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.h() == v.size());
		const V* ptr = m.ptr();
		V2* res = v.ptr();
		for(int i=0; i<m.h();i++) {
			int idx = 0;
			V val = *ptr;
			for(int j=0; j<m.w();j++) {
				if(*ptr > val) {
					val = *ptr;
					idx = j;
				}
				ptr++;
			}
			*res++ = idx;
		}
	}

}// namespace argmax_to_xxx
template<class V, class M>
void argmax_to_column(V&v, const M& m) {
	argmax_to_XXX_impl::argmax_to_column(v,m);
}

template<class V, class M>
void argmax_to_row(V&v, const M& m) {
	argmax_to_XXX_impl::argmax_to_row(v,m);
}

#define INSTANTIATE_ARGMAX_TO_ROW(V,M,I) \
  template void argmax_to_row(vector<int,dev_memory_space,I>&,const dense_matrix<V,M,dev_memory_space,I>&);   \
  template void argmax_to_row(vector<int,host_memory_space,I>&,const dense_matrix<V,M,host_memory_space,I>&);  \
  template void argmax_to_row(vector<float,dev_memory_space,I>&,const dense_matrix<V,M,dev_memory_space,I>&);   \
  template void argmax_to_row(vector<float,host_memory_space,I>&,const dense_matrix<V,M,host_memory_space,I>&);  
#define INSTANTIATE_ARGMAX_TO_COL(V,M,I) \
  template void argmax_to_column(vector<int,dev_memory_space,I>&,const dense_matrix<V,M,dev_memory_space,I>&);   \
  template void argmax_to_column(vector<int,host_memory_space,I>&,const dense_matrix<V,M,host_memory_space,I>&); \
  template void argmax_to_column(vector<float,dev_memory_space,I>&,const dense_matrix<V,M,dev_memory_space,I>&);   \
  template void argmax_to_column(vector<float,host_memory_space,I>&,const dense_matrix<V,M,host_memory_space,I>&);   

#define INSTANTIATE_REDCOL(V,M) \
  template void reduce_to_row(vector<V,dev_memory_space>&, const dense_matrix<V,M,dev_memory_space>&, reduce_functor,  const V&,const V&); \
  template void reduce_to_col(vector<V,dev_memory_space>&, const dense_matrix<V,M,dev_memory_space>&, reduce_functor, const V&,const V&); \
  template void reduce_to_row(vector<V,host_memory_space>&, const dense_matrix<V,M,host_memory_space>&, reduce_functor,  const V&,const V&); \
  template void reduce_to_col(vector<V,host_memory_space>&, const dense_matrix<V,M,host_memory_space>&,reduce_functor,  const V&,const V&);

#define INSTANTIATE_REDROW(V,M) \
  template void reduce_to_col(vector<V,dev_memory_space>&, const dense_matrix<V,M,dev_memory_space>&, reduce_functor, const V&,const V&); \
  template void reduce_to_row(vector<V,dev_memory_space>&, const dense_matrix<V,M,dev_memory_space>&, reduce_functor,  const V&,const V&); \
  template void reduce_to_col(vector<V,host_memory_space>&, const dense_matrix<V,M,host_memory_space>&, reduce_functor, const V&,const V&); \
  template void reduce_to_row(vector<V,host_memory_space>&, const dense_matrix<V,M,host_memory_space>&, reduce_functor,  const V&,const V&);

INSTANTIATE_ARGMAX_TO_COL(float,row_major,unsigned int);
INSTANTIATE_ARGMAX_TO_COL(int,row_major,unsigned int);

INSTANTIATE_ARGMAX_TO_ROW(float,column_major,unsigned int);
INSTANTIATE_ARGMAX_TO_ROW(int,column_major,unsigned int);

INSTANTIATE_REDCOL(float,column_major);
INSTANTIATE_REDROW(float,row_major);
};//namespace cuv
