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

#include <cuv/tools/cuv_general.hpp>
#include <cuv/tools/meta_programming.hpp>
#include <cuv/tensor_ops/functors.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>

template<int BLOCK_DIM, class T, class V, class RF>
__global__
void reduce_to_col_kernel(const T* matrix, V* vector, const unsigned int nCols, const unsigned int nRows,
		const T factNew, const T factOld, RF rf, const T init_value) {
	// reduce to column for column major matrices, reduce to row for row major matrices

	typedef cuv::reduce_functor_traits<typename RF::result_value_functor_type> functor_traits;
	typedef typename cuv::unconst<T>::type unconst_value_type;

	extern __shared__ unsigned char ptr[]; // need this intermediate variable for nvcc :-(
	unconst_value_type* values = (unconst_value_type*) ptr;
	unsigned int* indices = (unsigned int*)(values + BLOCK_DIM*BLOCK_DIM);
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
		T f = matrix[my * nRows + row_idx ];
		rf.rv(sum,arg_index,f,my);
		//sum=rf(sum,f);
	}

	values[ty*BLOCK_DIM+tx] = sum;
	if(functor_traits::returns_index)
		indices[ty*BLOCK_DIM+tx] = arg_index;

	__syncthreads();

	for (unsigned int offset = blockDim.y / 2; offset > 0; offset >>=1) {
		if (ty < offset) {
			const unsigned int v = ty+offset;
			rf.rr(
					  values [ty*BLOCK_DIM+tx],
					  indices[ty*BLOCK_DIM+tx],
					  values [v *BLOCK_DIM+tx],
					  indices[v *BLOCK_DIM+tx]);
		}
		__syncthreads();
	}
	
	if (ty == 0) {
		if (functor_traits::returns_index)
			vector[row_idx] = indices[tx];
		else
			if(factOld != 0.f){
				vector[row_idx] = vector[row_idx] * factOld + values[tx] * factNew;
			}else{
				vector[row_idx] = values[tx] * factNew;
			}
	}
}

template<int BLOCK_DIM, class T, class V, class RF>
__global__
void reduce_to_row_kernel(const T* matrix, V* vector, const unsigned int nCols, const unsigned int nRows,
		const T factNew, const T factOld, RF rf, const T init_value) {
	// reduce to row for column major matrices, reduce to column for row major matrices
	typedef cuv::reduce_functor_traits<typename RF::result_value_functor_type> functor_traits;
	typedef typename cuv::unconst<T>::type unconst_value_type;

	extern __shared__ float sptr[]; // need this intermediate variable for nvcc :-(
	unconst_value_type* values = (unconst_value_type*) sptr;
	unsigned int* indices                  = (unsigned int*)(values + BLOCK_DIM*BLOCK_DIM);
	const unsigned int tx = threadIdx.x; // blockIdx.x is always 0
	const unsigned int by = blockIdx.y; //threadIdx.y is always 0, blockDim.y is always 1!
	const unsigned int off = blockDim.x;
	
	values[tx] = init_value;
	if(functor_traits::returns_index)
		indices[tx] = 0;

	for (unsigned int my = tx; my < nRows; my += off) {
		const T f = matrix[by * nRows + my];
		rf.rv(values[tx],indices[tx],f,my);
	}
	__syncthreads();

	for (unsigned int offset = BLOCK_DIM*BLOCK_DIM/2; offset > 0; offset>>=1) {
		const unsigned int v = tx+offset;
		if (tx < offset)
			rf.rr(values[tx],indices[tx],values[v],indices[v]);
		__syncthreads();
	}
	__syncthreads();
	if (tx == 0) {
		if (functor_traits::returns_index)
			vector[by] = indices[0];
		else{
			if(factOld != 0){
				vector[by] = vector[by]
					* factOld + values[0] * factNew;
			}else{
				vector[by] = values[0] * factNew;
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
	template<int dim, class __memory_space_type>
	       struct reduce{};

	template<>
	struct reduce<1, dev_memory_space>{
                template<class __value_type, class __value_type2, class __memory_layout_type, class RF, class S>
	       	void operator()(tensor<__value_type,dev_memory_space> &v,const  dense_matrix<__value_type2,dev_memory_space,__memory_layout_type> &m,const  S & factNew,const  S & factOld, RF rf)const{
                    cuvAssert(m.ptr() != NULL);
                    cuvAssert(m.h() == v.size());
                    static const int BLOCK_DIM = 16;
                    const int blocks_needed = ceil((float)m.h()/(BLOCK_DIM));
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
                    dim3 threads(BLOCK_DIM,BLOCK_DIM);
                    typedef __value_type matval_t;
                    typedef typename tensor<__value_type,dev_memory_space>::value_type vecval_t;
                    unsigned int mem = sizeof(matval_t) * BLOCK_DIM*BLOCK_DIM ;

                    typedef reduce_functor_traits<typename RF::result_value_functor_type> traits_type;
                    if(traits_type::returns_index)
                            mem += sizeof(vecval_t)*BLOCK_DIM*BLOCK_DIM;
                    reduce_to_col_kernel<BLOCK_DIM><<<grid,threads,mem>>>(m.ptr(),v.ptr(),m.w(),m.h(),__value_type2(factNew),__value_type2(factOld),rf,__value_type2(traits_type::init_value()));
                    cuvSafeCall(cudaThreadSynchronize());
	}};

	template<>
	struct reduce<0, dev_memory_space>{
                template<class __value_type, class __value_type2, class __memory_layout_type, class RF, class S>
	       	void operator()(tensor<__value_type,dev_memory_space> &v,const  dense_matrix<__value_type2,dev_memory_space,__memory_layout_type> &m,const S & factNew,const  S & factOld, RF rf)const{
		cuvAssert(m.ptr() != NULL);
		cuvAssert(m.w() == v.size());
		static const int BLOCK_DIM = 16;
		dim3 grid(1, m.w());
		dim3 threads(BLOCK_DIM*BLOCK_DIM,1);

		typedef __value_type matval_t;
		typedef typename tensor<__value_type,dev_memory_space>::value_type vecval_t;
		unsigned int mem = sizeof(matval_t) * threads.x*threads.y;
		typedef reduce_functor_traits<typename RF::result_value_functor_type> traits_type;
		if(traits_type::returns_index)
			mem += sizeof(vecval_t)*threads.x*threads.y;

                reduce_to_row_kernel<BLOCK_DIM><<<grid,threads,mem>>>(m.ptr(),v.ptr(),m.w(),m.h(),__value_type2(factNew),__value_type2(factOld),rf,__value_type2(traits_type::init_value()));
		cuvSafeCall(cudaThreadSynchronize());
	}};

	template<int dim>
	struct reduce<dim, host_memory_space>{
                template<class __value_type, class __value_type2, class __memory_layout_type, class RF, class S>
	       	void operator()(tensor<__value_type,host_memory_space> &v,const  dense_matrix<__value_type2,host_memory_space,__memory_layout_type> &m,const S & factNew,const S & factOld, RF rf)const{
		typedef __value_type2 V;
		typedef typename tensor<__value_type,host_memory_space>::value_type V2;
		typedef typename dense_matrix<__value_type,host_memory_space,__memory_layout_type>::index_type I;
		typedef typename unconst<V>::type unconstV;
		typedef cuv::reduce_functor_traits<typename RF::result_value_functor_type> functor_traits;

		cuvAssert(m.ptr() != NULL);
		// assert that vector has correct length
		if (dim==0) cuvAssert(v.size()==m.w());
		if (dim==1) cuvAssert(v.size()==m.h());

		const __value_type2 * A_ptr                         = m.ptr();

		// indices: only needed when arg-max/arg-min etc used
		tensor<I,host_memory_space>* indices = NULL;
		I* indices_ptr                         = NULL;
		if(functor_traits::returns_index){
			indices         =  new tensor<I,host_memory_space>(v.size());
			indices_ptr     =  indices->ptr();
			memset(indices_ptr,indices->memsize(), 0);
		}
		I*const indices_begin = indices_ptr;
		I*const indices_end   = indices_ptr + v.size();

		// values: the new values that are to be combined with v using fact
		tensor<unconstV,host_memory_space> values(v.size());
		unconstV* values_ptr                   = values.ptr();
		V*const values_end                     = values_ptr + values.size();
		while(values_ptr != values_end) 
			*values_ptr++ =functor_traits::init_value(); 
		values_ptr = values.ptr();      // reset pointers to begining of vector

		if (dim==0){
			// apply reduce functor along columns
			for(;values_ptr!=values_end; values_ptr++, indices_ptr++) {
				for(unsigned int j=0; j<m.h(); j++, A_ptr++)
					rf.rv(*values_ptr,*indices_ptr,*A_ptr,j);
			}
		}
		else if(dim==1){
			// apply reduce functor along rows
			for(I i=0;i<m.w();i++) {
				values_ptr  = values.ptr();
				indices_ptr = indices_begin;
				for(; values_ptr!=values_end;A_ptr++,values_ptr++,indices_ptr++) 
					rf.rv(*values_ptr,*indices_ptr,*A_ptr,i);
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

        template<int dimension, class __value_type, class __value_type2, class __memory_space_type, class __memory_layout_type, class S>
	void reduce_switch(tensor<__value_type,__memory_space_type>&v, const dense_matrix<__value_type2,__memory_space_type,__memory_layout_type>& m, reduce_functor rf, const S& factNew, const S& factOld) {
		typedef __value_type2 const_mat_val;
		typedef typename dense_matrix<__value_type2,__memory_space_type,__memory_layout_type>::index_type mat_ind;
		typedef typename dense_matrix<__value_type2,__memory_space_type,__memory_layout_type>::memory_space_type mat_mem;
		typedef typename tensor<__value_type,__memory_space_type>::value_type vec_val;
		typedef typename tensor<__value_type,__memory_space_type>::index_type vec_ind;
		typedef typename unconst<const_mat_val>::type mat_val;
		switch(rf) {
			case RF_ADD:
			reduce_impl::reduce<dimension,mat_mem>()(v,m,factNew,factOld,make_reduce_functor(bf_plus<vec_val,vec_val,mat_val>(),bf_plus<vec_val,vec_val,vec_val>()));
			break;
			case RF_ADD_SQUARED:
			reduce_impl::reduce<dimension,mat_mem>()(v,m,factNew,factOld,make_reduce_functor(bf_add_square<vec_val,vec_val,mat_val>(),bf_plus<vec_val,vec_val,vec_val>()));
			break;
			case RF_MIN:
			reduce_impl::reduce<dimension,mat_mem>()(v,m,factNew,factOld,make_reduce_functor(bf_min<mat_val,mat_val,mat_val>()));
			break;
			case RF_MAX:
			reduce_impl::reduce<dimension,mat_mem>()(v,m,factNew,factOld,make_reduce_functor(bf_max<mat_val,mat_val,mat_val>()));
			break;
			case RF_ARGMAX:
			reduce_impl::reduce<dimension,mat_mem>()(v,m,factNew,factOld,make_arg_reduce_functor(reduce_argmax<mat_val,mat_ind>()));
			break;
			case RF_ARGMIN:
			reduce_impl::reduce<dimension,mat_mem>()(v,m,factNew,factOld,make_arg_reduce_functor(reduce_argmin<mat_val,mat_ind>()));
			break;
			case RF_MULT:
			reduce_impl::reduce<dimension,mat_mem>()(v,m,factNew,factOld,make_reduce_functor(bf_add_log<mat_val,mat_val,mat_val>(), bf_plus<vec_val,vec_val,mat_val>()));
			apply_scalar_functor(v,SF_EXP);
			break;
			case RF_LOGADDEXP:
			reduce_impl::reduce<dimension,mat_mem>()(v,m,factNew,factOld,make_reduce_functor(bf_logaddexp<mat_val>()));
			break;
			case RF_ADDEXP:
			reduce_impl::reduce<dimension,mat_mem>()(v,m,factNew,factOld,make_reduce_functor(bf_logaddexp<mat_val>()));
			apply_scalar_functor(v,SF_EXP);
			break;
			default:
			throw std::runtime_error("supplied reduce_functor is not implemented");
		}
	}


}//namespace reduce_imp

// TODO: make sure this is actually called with a matrix type!
//
template<class __value_type, class __value_type2, class __memory_space_type, class __memory_layout_type>
void reduce_to_col(tensor<__value_type,__memory_space_type>&v, const dense_matrix<__value_type2,__memory_space_type,__memory_layout_type>& m, reduce_functor rf, const __value_type2& factNew, const __value_type2& factOld) {
	if (IsSame<typename dense_matrix<__value_type,__memory_space_type,__memory_layout_type>::memory_layout,row_major>::Result::value){
		//matrix is row major
		//create column major view and call reduce_to_row for column major
		// downstream from here everything is column major
		const dense_matrix<const __value_type2,typename dense_matrix<__value_type,__memory_space_type,__memory_layout_type>::memory_space_type,column_major,typename dense_matrix<__value_type,__memory_space_type,__memory_layout_type>::index_type> cm_view(m.w(),m.h(),m.ptr());
		reduce_impl::reduce_switch<0>(v,cm_view,rf,factNew,factOld); // 0 means zeroth dimension is summed out - meaning summing over the columns in a column major matrix.
	}
	else {
		reduce_impl::reduce_switch<1>(v,m,rf,factNew,factOld); // 1 means first dimension (we start counting at zero) is summed out - meaning summing over the rows in a column major matrix.
	}
}

template<class __value_type, class __value_type2, class __memory_space_type, class __memory_layout_type>
void reduce_to_row(tensor<__value_type,__memory_space_type>&v, const dense_matrix<__value_type2,__memory_space_type,__memory_layout_type>& m,reduce_functor rf, const __value_type2& factNew, const __value_type2& factOld) {
	if (IsSame<typename dense_matrix<__value_type,__memory_space_type,__memory_layout_type>::memory_layout,row_major>::Result::value){
		//matrix is row major
		//create column major view and call reduce_to_row for column major
		// downstream from here everything is column major
		const dense_matrix<const __value_type2,__memory_space_type,column_major> cm_view(m.w(),m.h(),m.ptr());
		reduce_impl::reduce_switch<1>(v,cm_view,rf,factNew,factOld); // 1 means first (we start counting at zero) dimension is summed out - meaning summing over the rows in a column major matrix.
	}
	else {
		reduce_impl::reduce_switch<0>(v,m,rf,factNew,factOld); // 0 means zeroth dimension is summed out - meaning summing over the columns in a column major matrix.
	}
	
}

/*namespace argmax_to_XXX_impl{*/
	/*template<class V, class V2, class I>*/
	/*void argmax_to_row(tensor<V2,dev_memory_space>&v, const dense_matrix<V, dev_memory_space,column_major, I>& m) {*/
		/*cuvAssert(m.ptr() != NULL);*/
		/*cuvAssert(m.w() == v.size());*/
		/*const unsigned int u = min(m.w(), MAX_GRID_SIZE);*/
		/*dim3 grid(u, ceil(m.w()/(float)u));*/
		/*static const unsigned int BLOCK_DIM = 256;*/
		/*argmax_row_kernel<BLOCK_DIM><<<grid,BLOCK_DIM>>>(v.ptr(),m.ptr(),m.w(),m.h());*/
		/*cuvSafeCall(cudaThreadSynchronize());*/
	/*}*/

	/*template<class V, class V2, class I>*/
	/*void argmax_to_column(tensor<V2,dev_memory_space>&v, const dense_matrix<V,row_major,dev_memory_space,I>& m) {*/
		/*cuvAssert(m.ptr() != NULL);*/
		/*cuvAssert(m.h() == v.size());*/
		/*const unsigned int u = min(m.h(), MAX_GRID_SIZE);*/
		/*dim3 grid(u, ceil(m.h()/(float)u));*/
		/*static const unsigned int BLOCK_DIM = 256;*/
		/*argmax_row_kernel<BLOCK_DIM><<<grid,BLOCK_DIM>>>(v.ptr(),m.ptr(),m.h(),m.w());*/
		/*cuvSafeCall(cudaThreadSynchronize());*/
	/*}*/

	/*template<class V, class V2, class I>*/
	/*void argmax_to_row(tensor<V2,host_memory_space>&v, const dense_matrix<V, host_memory_space,column_major,I>& m) {*/
		/*cuvAssert(m.ptr() != NULL);*/
		/*cuvAssert(m.w() == v.size());*/
		/*const V* ptr = m.ptr();*/
		/*V2* res = v.ptr();*/
		/*for(int i=0; i<m.w();i++) {*/
			/*int idx = 0;*/
			/*V val = *ptr;*/
			/*for(int j=0; j<m.h();j++) {*/
				/*if(*ptr > val) {*/
					/*val = *ptr;*/
					/*idx = j;*/
				/*}*/
				/*ptr++;*/
			/*}*/
			/**res++ = idx;*/
		/*}*/
	/*}*/

	/*template<class V, class V2, class I>*/
	/*void argmax_to_column(tensor<V2,host_memory_space>&v, const dense_matrix<V,row_major,host_memory_space,I>& m) {*/
		/*cuvAssert(m.ptr() != NULL);*/
		/*cuvAssert(m.h() == v.size());*/
		/*const V* ptr = m.ptr();*/
		/*V2* res = v.ptr();*/
		/*for(int i=0; i<m.h();i++) {*/
			/*int idx = 0;*/
			/*V val = *ptr;*/
			/*for(int j=0; j<m.w();j++) {*/
				/*if(*ptr > val) {*/
					/*val = *ptr;*/
					/*idx = j;*/
				/*}*/
				/*ptr++;*/
			/*}*/
			/**res++ = idx;*/
		/*}*/
	/*}*/

/*}// namespace argmax_to_xxx*/
/*template<class V, class M>*/
/*void argmax_to_column(V&v, const M& m) {*/
	/*argmax_to_XXX_impl::argmax_to_column(v,m);*/
/*}*/

/*template<class V, class M>*/
/*void argmax_to_row(V&v, const M& m) {*/
	/*argmax_to_XXX_impl::argmax_to_row(v,m);*/
/*}*/

/*#define INSTANTIATE_ARGMAX_TO_ROW(V,M,I) \*/
  /*template void argmax_to_row(tensor<int,dev_memory_space>&,const dense_matrix<V,dev_memory_space,M,I>&);   \*/
  /*template void argmax_to_row(tensor<int,host_memory_space>&,const dense_matrix<V,host_memory_space,M,I>&);  \*/
  /*template void argmax_to_row(tensor<float,dev_memory_space>&,const dense_matrix<V,dev_memory_space,M,I>&);   \*/
  /*template void argmax_to_row(tensor<float,host_memory_space>&,const dense_matrix<V,host_memory_space,M,I>&);  */
/*#define INSTANTIATE_ARGMAX_TO_COL(V,M,I) \*/
  /*template void argmax_to_column(tensor<int,dev_memory_space>&,const dense_matrix<V,dev_memory_space,M,I>&);   \*/
  /*template void argmax_to_column(tensor<int,host_memory_space>&,const dense_matrix<V,host_memory_space,M,I>&); \*/
  /*template void argmax_to_column(tensor<float,dev_memory_space>&,const dense_matrix<V,dev_memory_space,M,I>&);   \*/
  /*template void argmax_to_column(tensor<float,host_memory_space>&,const dense_matrix<V,host_memory_space,M,I>&);   */

#define INSTANTIATE_RED(V,V2,M) \
  template void reduce_to_row(tensor<V2,dev_memory_space>&, const dense_matrix<V,dev_memory_space,M>&, reduce_functor,  const V&,const V&); \
  template void reduce_to_col(tensor<V2,dev_memory_space>&, const dense_matrix<V,dev_memory_space,M>&, reduce_functor, const V&,const V&); \
  template void reduce_to_row(tensor<V2,host_memory_space>&, const dense_matrix<V,host_memory_space,M>&, reduce_functor,  const V&,const V&); \
  template void reduce_to_col(tensor<V2,host_memory_space>&, const dense_matrix<V,host_memory_space,M>&,reduce_functor,  const V&,const V&);


/*INSTANTIATE_ARGMAX_TO_COL(float,row_major,unsigned int);*/
/*INSTANTIATE_ARGMAX_TO_COL(int,row_major,unsigned int);*/

/*INSTANTIATE_ARGMAX_TO_ROW(float,column_major,unsigned int);*/
/*INSTANTIATE_ARGMAX_TO_ROW(int,column_major,unsigned int);*/

INSTANTIATE_RED(float,float,column_major);
/*INSTANTIATE_RED(int,float,column_major);*/
/*INSTANTIATE_RED(unsigned int,float,column_major);*/
/*INSTANTIATE_RED(unsigned char,float,column_major);*/
/*INSTANTIATE_RED(float,int,column_major);*/
INSTANTIATE_RED(float,unsigned int,column_major);
/*INSTANTIATE_RED(float,unsigned char,column_major);*/
INSTANTIATE_RED(unsigned char,unsigned char,column_major);
/*INSTANTIATE_RED(unsigned char,unsigned int,column_major);*/

INSTANTIATE_RED(float,float,row_major);
/*INSTANTIATE_RED(int,float,row_major);*/
/*INSTANTIATE_RED(unsigned int,float,row_major);*/
/*INSTANTIATE_RED(unsigned char,float,row_major);*/
/*INSTANTIATE_RED(float,int,row_major);*/
INSTANTIATE_RED(float,unsigned int,row_major);
/*INSTANTIATE_RED(float,unsigned char,row_major);*/
INSTANTIATE_RED(unsigned char,unsigned char,row_major);
/*INSTANTIATE_RED(unsigned char,unsigned int,row_major);*/
};//namespace cuv

