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
void reduce_to_col_kernel(const T* matrix, V* vector, const unsigned int nCols, const unsigned int nRows,
		const T factNew, const T factOld, RF reduce_functor, const T init_value) {

	typedef typename cuv::reduce_functor_traits<RF> functor_traits;
	typedef typename cuv::functor_dispatcher<functor_traits::returns_index> functor_dispatcher_type;
	typedef typename cuv::unconst<T>::type unconst_value_type;
	functor_dispatcher_type func_disp;

	extern __shared__ unsigned char ptr[]; // need this intermediate variable for nvcc :-(
	unconst_value_type* values = (unconst_value_type*) ptr;
	unsigned int* indices = (unsigned int*)(values + BLOCK_SIZE*BLOCK_SIZE);
	const unsigned int tx = threadIdx.x;
	const unsigned int bx = blockIdx.x;
	const unsigned int ty = threadIdx.y;
	const unsigned int by = blockIdx.y;

	const int row_idx = by * gridDim.x * blockDim.x +   	// offset according to y index in grid
						bx * blockDim.x +  					// offset according to block index
						tx;									// offset in block

	if (row_idx >= nRows)
		return;
	const unsigned int off = blockDim.y;

	unconst_value_type sum = init_value;
	unsigned int arg_index = 0; // for storing indeces of maxima/minima for arg functors

	for (unsigned int my = ty; my < nCols; my += off) {
		const T f = matrix[my * nRows + row_idx ];
		func_disp(reduce_functor,sum,arg_index,f,my);
		//sum=reduce_functor(sum,f);
	}

	values[ty*BLOCK_SIZE+tx] = sum;
	if(functor_traits::returns_index)
		indices[ty*BLOCK_SIZE+tx] = arg_index;

	__syncthreads();

	for (unsigned int offset = blockDim.y / 2; offset > 0; offset >>=1) {
		if (ty < offset) {
			const unsigned int v = ty+offset;
			func_disp(reduce_functor,
					  values [ty*BLOCK_SIZE+tx],
					  indices[ty*BLOCK_SIZE+tx],
					  values [v *BLOCK_SIZE+tx],
					  indices[v *BLOCK_SIZE+tx]);
		}
		__syncthreads();
	}
	
	if (ty == 0) {
		if (cuv::reduce_functor_traits<RF>::returns_index)
			vector[row_idx] = indices[tx];
		else
			if(factOld != 0.f){
				vector[row_idx] = vector[row_idx] * factOld + values[tx] * factNew;
			}else{
				vector[row_idx] = values[tx] * factNew;
			}
	}
}

template<int BLOCK_SIZE, class T, class V, class RF>
__global__
void reduce_to_row_kernel(const T* matrix, V* vector, const unsigned int nCols, const unsigned int nRows,
		const T factNew, const T factOld, const RF reduce_functor, const T init_value) {

	typedef typename cuv::reduce_functor_traits<RF> functor_traits;
	typedef typename cuv::functor_dispatcher<functor_traits::returns_index> functor_dispatcher_type;
	typedef typename cuv::unconst<T>::type unconst_value_type;
	functor_dispatcher_type func_disp;

	extern __shared__ float sptr[]; // need this intermediate variable for nvcc :-(
	unconst_value_type* values = (unconst_value_type*) sptr;
	unsigned int* indices                  = (unsigned int*)(values + BLOCK_SIZE*BLOCK_SIZE);
	const unsigned int tx = threadIdx.x, bx = blockIdx.x;
	const unsigned int ty = threadIdx.y, by = blockIdx.y;
	const unsigned int off = blockDim.x;
	
	values[tx] = init_value;
	if(functor_traits::returns_index)
		indices[tx] = 0;

	for (unsigned int my = tx; my < nRows; my += off) {
		const T f = matrix[by * nRows + bx * blockDim.x + my];
		func_disp(reduce_functor,values[tx],indices[tx],f,my);
	}
	__syncthreads();

	for (unsigned int offset = BLOCK_SIZE*BLOCK_SIZE/2; offset > 0; offset>>=1) {
		const unsigned int v = tx+offset;
		if (tx < offset)
			func_disp(reduce_functor,values[tx],indices[tx],values[v],indices[v]);
		__syncthreads();
	}
	__syncthreads();
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

namespace reduce_impl {
	template<int dim, class __memory_space_type, class __matrix_type, class __vector_type, class RF>
		struct reduce{ void operator()(__vector_type &v, const __matrix_type &m, const typename __matrix_type::value_type & factNew, const typename __matrix_type::value_type & factOld, RF reduce_functor)const{
			cuvAssert(false);
		}};

	template<class __matrix_type, class __vector_type, class RF>
	struct reduce<1, dev_memory_space, __matrix_type, __vector_type, RF>{ void operator()(__vector_type &v,const  __matrix_type &m,const  typename __matrix_type::value_type & factNew,const  typename __matrix_type::value_type & factOld, RF reduce_functor)const{
	//void reduce<1>(vector<V2,dev_memory_space,I>&v, const dense_matrix<V,column_major,dev_memory_space,I>& m, const V& factNew, const V& factOld, RF reduce_functor) {
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.h() == v.size());
		static const int BLOCK_SIZE = 16;
		static const int BLOCK_DIM_X = BLOCK_SIZE;
		static const int BLOCK_DIM_Y = BLOCK_SIZE;
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
		typedef typename __matrix_type::value_type matval_t;
		typedef typename __vector_type::value_type vecval_t;
		unsigned int mem = sizeof(matval_t) * BLOCK_DIM_X*BLOCK_DIM_Y ;
		if(reduce_functor_traits<RF>::returns_index)
			mem += sizeof(vecval_t)*BLOCK_DIM_X*BLOCK_DIM_Y;
		reduce_to_col_kernel<BLOCK_SIZE,matval_t><<<grid,threads,mem>>>(m.ptr(),v.ptr(),m.w(),m.h(),factNew,factOld,reduce_functor,reduce_functor_traits<RF>::init_value());
		cuvSafeCall(cudaThreadSynchronize());
	}};

	template<class __matrix_type, class __vector_type, class RF>
	struct reduce<0, dev_memory_space, __matrix_type, __vector_type, RF>{ void operator()(__vector_type &v,const  __matrix_type &m,const  typename __matrix_type::value_type & factNew,const  typename __matrix_type::value_type & factOld, RF reduce_functor)const{
	//void reduce<0>(vector<V2,dev_memory_space,I>&v, const dense_matrix<V,column_major,dev_memory_space,I>& m, const V& factNew, const V& factOld, RF reduce_functor) {
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.w() == v.size());
		static const int BLOCK_SIZE = 16;
		dim3 grid(1, m.w());
		dim3 threads(BLOCK_SIZE*BLOCK_SIZE,1);

		typedef typename __matrix_type::value_type matval_t;
		typedef typename __vector_type::value_type vecval_t;
		unsigned int mem = sizeof(matval_t) * threads.x*threads.y;
		if(reduce_functor_traits<RF>::returns_index)
			mem += sizeof(vecval_t)*threads.x*threads.y;

		reduce_to_row_kernel<BLOCK_SIZE,matval_t><<<grid,threads,mem>>>(m.ptr(),v.ptr(),m.w(),m.h(),factNew,factOld,reduce_functor,reduce_functor_traits<RF>::init_value());
		cuvSafeCall(cudaThreadSynchronize());
	}};

	template<int dim, class __matrix_type, class __vector_type, class RF>
	//struct reduce<dim, host_memory_space, __matrix_type, __vector_type, RF>{ void operator()(vector<V2,host_memory_space,I>&v, const dense_matrix<V,column_major,host_memory_space,I>& m, const V& factNew, const V& factOld, RF reduce_functor) {
	struct reduce<dim, host_memory_space, __matrix_type, __vector_type, RF>{ void operator()(__vector_type&v, const __matrix_type & m, const typename __matrix_type::value_type& factNew, const typename __matrix_type::value_type& factOld, RF reduce_functor) const{
		typedef typename __matrix_type::value_type V;
		typedef typename __vector_type::value_type V2;
		typedef typename __matrix_type::index_type I;
		typedef typename unconst<V>::type unconstV;
		typedef typename cuv::reduce_functor_traits<RF> functor_traits;
		typedef typename cuv::functor_dispatcher<functor_traits::returns_index> functor_dispatcher_type;

		cuvAssert(m.ptr() != NULL);
		// assert that vector has correct length
		if (dim==0) cuvAssert(v.size()==m.w());
		if (dim==1) cuvAssert(v.size()==m.h());

		functor_dispatcher_type func_disp;
		const V* A_ptr                         = m.ptr();

		// indices: only needed when arg-max/arg-min etc used
		vector<I,host_memory_space,I>* indices = NULL;
		I* indices_ptr                         = NULL;
		if(functor_traits::returns_index){
			indices         =  new vector<I,host_memory_space,I>(v.size());
			indices_ptr     =  indices->ptr();
			memset(indices_ptr,indices->memsize(), 0);
		}
		I*const indices_begin = indices_ptr;
		I*const indices_end   = indices_ptr + v.size();

		// values: the new values that are to be combined with v using factNew/factOld
		vector<unconstV,host_memory_space,I> values(v.size());
		unconstV* values_ptr                   = values.ptr();
		V*const values_end                     = values_ptr + values.size();
		while(values_ptr != values_end) 
			*values_ptr++ =functor_traits::init_value(); 
		values_ptr = values.ptr();      // reset pointers to begining of vector

		if (dim==0){
			// apply reduce functor along columns
			for(;values_ptr!=values_end; values_ptr++, indices_ptr++) {
				for(unsigned int j=0; j<m.h(); j++, A_ptr++)
					func_disp(reduce_functor,*values_ptr,*indices_ptr,*A_ptr,j);
			}
		}
		else if(dim==1){
			// apply reduce functor along rows
			for(int i=0;i<m.w();i++) {
				values_ptr  = values.ptr();
				indices_ptr = indices_begin;
				for(; values_ptr!=values_end;A_ptr++,values_ptr++,indices_ptr++) 
					func_disp(reduce_functor,*values_ptr,*indices_ptr,*A_ptr,i);
			}
		}else{
			cuvAssert(false);
		}

		// reset pointers to begining of vectors
		values_ptr  = values.ptr();
		indices_ptr = indices_begin;

		// put result into v via v_ptr.
		V2* v_ptr   = v.ptr();
		if (!functor_traits::returns_index){ 
			if (factOld!=0){
				while(values_ptr!=values_end) 
					*v_ptr   = factOld * *v_ptr++  + factNew * *values_ptr++;
			}else
				while(values_ptr!=values_end) 
					*v_ptr++ = factNew * *values_ptr++;
		}
		else{
			while(indices_ptr!=indices_end) 
				*v_ptr++ = *indices_ptr++;
			delete indices;
		}
	}};

	template<int dimension, class __matrix_type, class __vector_type>
	void reduce_switch(__vector_type&v, const __matrix_type& m, reduce_functor rf, const typename __matrix_type::value_type& factNew, const typename __matrix_type::value_type& factOld) {
		typedef typename __matrix_type::value_type mat_val;
		typedef typename __matrix_type::index_type mat_ind;
		typedef typename __matrix_type::memory_space_type mat_mem;
		typedef typename __vector_type::value_type vec_val;
		typedef typename __vector_type::index_type vec_ind;
		typedef typename unconst<mat_val>::type unconst_mat_val;
		switch(rf) {
			case RF_ADD:
			reduce_impl::reduce<dimension,mat_mem,__matrix_type,__vector_type,bf_plus<mat_val,mat_val> >() (v,m,factNew,factOld,bf_plus<mat_val,mat_val>());
			break;
			case RF_ADD_SQUARED:
			reduce_impl::reduce<dimension,mat_mem,__matrix_type,__vector_type,bf_add_square<mat_val,mat_val> >()(v,m,factNew,factOld,bf_add_square<mat_val,mat_val>());
			break;
			case RF_MIN:
			reduce_impl::reduce<dimension,mat_mem,__matrix_type,__vector_type,bf_min<mat_val,mat_val> >()(v,m,factNew,factOld,bf_min<mat_val,mat_val>());
			break;
			case RF_MAX:
			reduce_impl::reduce<dimension,mat_mem,__matrix_type,__vector_type,bf_max<mat_val,mat_val> >()(v,m,factNew,factOld,bf_max<mat_val,mat_val>());
			break;
			case RF_ARGMAX:
			reduce_impl::reduce<dimension,mat_mem,__matrix_type,__vector_type,reduce_argmax<unconst_mat_val,mat_ind> >()(v,m,factNew,factOld,reduce_argmax<unconst_mat_val,mat_ind>());
			break;
			case RF_ARGMIN:
			reduce_impl::reduce<dimension,mat_mem,__matrix_type,__vector_type,reduce_argmin<unconst_mat_val,mat_ind> >()(v,m,factNew,factOld,reduce_argmin<unconst_mat_val,mat_ind>());
			break;
			case RF_MULT:
			reduce_impl::reduce<dimension,mat_mem,__matrix_type,__vector_type,bf_add_log<mat_val,mat_val> >()(v,m,factNew,factOld,bf_add_log<mat_val,mat_val>());
			apply_scalar_functor(v,SF_EXP);
			break;
			case RF_LOGADDEXP:
			reduce_impl::reduce<dimension,mat_mem,__matrix_type,__vector_type,bf_logaddexp<unconst_mat_val> >()(v,m,factNew,factOld,bf_logaddexp<unconst_mat_val>());
			break;
			case RF_ADDEXP:
			reduce_impl::reduce<dimension,mat_mem,__matrix_type,__vector_type,bf_logaddexp<unconst_mat_val> >()(v,m,factNew,factOld,bf_logaddexp<unconst_mat_val>());
			apply_scalar_functor(v,SF_EXP);
			break;
			default:
			throw std::runtime_error("supplied reduce_functor is not implemented");
		}
	}


}//namespace reduce_imp

// TODO: make sure this is actually called with a matrix type!
//
template<class __matrix_type, class __vector_type>
void reduce_to_col(__vector_type&v, const __matrix_type& m, reduce_functor rf, const typename __matrix_type::value_type& factNew, const typename __matrix_type::value_type& factOld) {
	if (IsSame<typename __matrix_type::memory_layout,row_major>::Result::value){
		//matrix is row major
		//create column major view and call reduce_to_row for column major
		// downstream from here everything is column major
		const dense_matrix<const typename __matrix_type::value_type,column_major,typename __matrix_type::memory_space_type,typename __matrix_type::index_type> cm_view(m.w(),m.h(),m.ptr(),true);
		reduce_impl::reduce_switch<0>(v,cm_view,rf,factNew,factOld); // 0 means zeroth dimension is summed out - meaning summing over the columns in a column major matrix.
	}
	else {
		reduce_impl::reduce_switch<1>(v,m,rf,factNew,factOld); // 1 means first dimension (we start counting at zero) is summed out - meaning summing over the rows in a column major matrix.
	}
}

template<class __matrix_type, class __vector_type>
void reduce_to_row(__vector_type&v, const __matrix_type& m,reduce_functor rf, const typename __matrix_type::value_type& factNew, const typename __matrix_type::value_type& factOld) {
	if (IsSame<typename __matrix_type::memory_layout,row_major>::Result::value){
		//matrix is row major
		//create column major view and call reduce_to_row for column major
		// downstream from here everything is column major
		const dense_matrix<const typename __matrix_type::value_type,column_major,typename __matrix_type::memory_space_type,typename __matrix_type::index_type> cm_view(m.w(),m.h(),m.ptr(),true);
		reduce_impl::reduce_switch<1>(v,cm_view,rf,factNew,factOld); // 1 means first (we start counting at zero) dimension is summed out - meaning summing over the rows in a column major matrix.
	}
	else {
		reduce_impl::reduce_switch<0>(v,m,rf,factNew,factOld); // 0 means zeroth dimension is summed out - meaning summing over the columns in a column major matrix.
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

#define INSTANTIATE_REDCOL(V,V2,M) \
  template void reduce_to_row(vector<V2,dev_memory_space>&, const dense_matrix<V,M,dev_memory_space>&, reduce_functor,  const V&,const V&); \
  template void reduce_to_col(vector<V2,dev_memory_space>&, const dense_matrix<V,M,dev_memory_space>&, reduce_functor, const V&,const V&); \
  template void reduce_to_row(vector<V2,host_memory_space>&, const dense_matrix<V,M,host_memory_space>&, reduce_functor,  const V&,const V&); \
  template void reduce_to_col(vector<V2,host_memory_space>&, const dense_matrix<V,M,host_memory_space>&,reduce_functor,  const V&,const V&);

#define INSTANTIATE_REDROW(V,V2,M) \
  template void reduce_to_col(vector<V2,dev_memory_space>&, const dense_matrix<V,M,dev_memory_space>&, reduce_functor, const V&,const V&); \
  template void reduce_to_row(vector<V2,dev_memory_space>&, const dense_matrix<V,M,dev_memory_space>&, reduce_functor,  const V&,const V&); \
  template void reduce_to_col(vector<V2,host_memory_space>&, const dense_matrix<V,M,host_memory_space>&, reduce_functor, const V&,const V&); \
  template void reduce_to_row(vector<V2,host_memory_space>&, const dense_matrix<V,M,host_memory_space>&, reduce_functor,  const V&,const V&);

INSTANTIATE_ARGMAX_TO_COL(float,row_major,unsigned int);
INSTANTIATE_ARGMAX_TO_COL(int,row_major,unsigned int);

INSTANTIATE_ARGMAX_TO_ROW(float,column_major,unsigned int);
INSTANTIATE_ARGMAX_TO_ROW(int,column_major,unsigned int);

INSTANTIATE_REDCOL(float,float,column_major);
INSTANTIATE_REDROW(float,float,row_major);
INSTANTIATE_REDCOL(float,int,column_major);
INSTANTIATE_REDROW(float,int,row_major);
INSTANTIATE_REDCOL(float,unsigned int,column_major);
INSTANTIATE_REDROW(float,unsigned int,row_major);
};//namespace cuv

