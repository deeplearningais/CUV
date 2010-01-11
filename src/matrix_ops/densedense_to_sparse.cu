#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include "densedense_to_sparse.hpp"
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]

using namespace std;

// multiply two dense matrices and put the result in an existing sparse DIA-formated matrix
template <class value_type, class index_type>                                                                        
__global__                                                                                                            
void                                                                                                                  
dense2dia_mm( value_type* C, const value_type* A, const value_type* B, index_type wA, index_type wB, index_type* blockidx, int dialen)
{
	uint2 blk = ((uint2*) blockidx)[SPARSE_DIA_BLOCK_SIZE_LEN/2 * blockIdx.x ];

	__shared__ index_type dia_offsets[SPARSE_DIA_BLOCK_SIZE*2];
	index_type v = __mul24(SPARSE_DIA_BLOCK_SIZE,threadIdx.y) + threadIdx.x;
	if(v < SPARSE_DIA_BLOCK_SIZE*2)
		dia_offsets[v] = blockidx[SPARSE_DIA_BLOCK_SIZE_LEN * blockIdx.x + 2 + v]; // 2: the two ints read already above

	__syncthreads();

    index_type tx = threadIdx.x;                                                                                             
    index_type ty = threadIdx.y;                                                                                             
                                                                                                                      
    index_type aBegin = wA * SPARSE_DIA_BLOCK_SIZE * blk.y;                                                              
    index_type aEnd   = aBegin + wA - 1;
    index_type aStep  = SPARSE_DIA_BLOCK_SIZE;                                                                                          
                                                                                                                      
    index_type bBegin = wA * SPARSE_DIA_BLOCK_SIZE * blk.x;                                                              
    index_type bStep  = SPARSE_DIA_BLOCK_SIZE;                                                                                          
                                                                                                                      
    value_type Csub = 0;                                                                                            
                                                                                                                      
    index_type waty = __mul24(wA,ty)+tx;                                                                                     
                                                                                                                      

    for (index_type a = aBegin, b  = bBegin;                                                                                 
             a <= aEnd;                                                                                               
             a += aStep, b += bStep) {                                                                                
                                                                                                                      
        __shared__ value_type As[SPARSE_DIA_BLOCK_SIZE][SPARSE_DIA_BLOCK_SIZE];                                                           
        __shared__ value_type Bs[SPARSE_DIA_BLOCK_SIZE][SPARSE_DIA_BLOCK_SIZE+1];                                                         
                                                                                                                      
		AS(ty, tx) = A[a + waty];                                                                     
		BS(ty, tx) = B[b + waty];                                                                     
                                                                                                                      
		__syncthreads();  // Synchronize to make sure the matrices are loaded                                                          
																													  
		for (index_type k = 0; k < SPARSE_DIA_BLOCK_SIZE; ++k)
		   Csub += AS(ty,k)*BS(tx, k);
		__syncthreads();
    }

	// diagonal in block
	int dia = tx - ty;
	int dia_real = __mul24(SPARSE_DIA_BLOCK_SIZE,blk.x) - __mul24(SPARSE_DIA_BLOCK_SIZE,blk.y) + dia;
	int dia_sparse = dia_offsets[SPARSE_DIA_BLOCK_SIZE-1+dia];
	int offd =  __mul24(blk.y,SPARSE_DIA_BLOCK_SIZE)+ty;
	/*printf("Csub=%2.2f dia=%d dia_real=%d dia_sparse=%d tx=%d ty=%d blk=%d,%d\n",Csub,dia,dia_real,dia_sparse,tx,ty,blk.x,blk.y);*/
	if(dia_sparse >= 0){
		int idx = dia_sparse*dialen           // the diagonal in the final matrix
		   + offd;                            // offset within diagonal
		/*printf("idx=%5d Csub=%2.2f dia=%d dia_real=%d dia_sparse=%d tx=%d ty=%d blk=%d,%d\n",idx,Csub,dia,dia_real,dia_sparse,tx,ty,blk.x,blk.y);*/
	   C[ idx ] += Csub;
	}
}   

namespace cuv{
	template<class V, class I>
		dev_block_descriptor<V,I>::dev_block_descriptor(const diamat_type& mat){
			thrust::host_vector<int> dia_offsets(
					thrust::device_ptr<const int>(mat.get_offsets().ptr()),
					thrust::device_ptr<const int>(mat.get_offsets().ptr()+mat.get_offsets().size()));
			std::vector<block> blocks;
			for(int i=0;i<mat.h();i+=SPARSE_DIA_BLOCK_SIZE){
				for(int j=0;j<mat.w();j+=SPARSE_DIA_BLOCK_SIZE){
					/*int upperdia = (j+SPARSE_DIA_BLOCK_SIZE-1) - i; // diagonal of upper right element of BLOCK_SIZExBLOCK_SIZE block*/
					int lowerdia = j - (i+SPARSE_DIA_BLOCK_SIZE-1); // diagonal of lower left  element of BLOCK_SIZExBLOCK_SIZE block
					block b;
					b.startx = j/SPARSE_DIA_BLOCK_SIZE;
					b.starty = i/SPARSE_DIA_BLOCK_SIZE;
					bool founddiag = false;
					for(int e=0; e<2*SPARSE_DIA_BLOCK_SIZE-1;e++){ // diag within block
						for(int d=0; d< dia_offsets.size();d++){
							if(lowerdia + e == dia_offsets[d]){
								b.diag[e] = d;
								founddiag = true;
								break; // look at next diag of block
							}else{
								b.diag[e] = -1;
							}
						}
					}
					if(founddiag){
						blocks.push_back(b);
					}
				}
			}
			cuvSafeCall(cudaMalloc((void**)&m_blocks.ptr, sizeof(int) * SPARSE_DIA_BLOCK_SIZE_LEN*blocks.size()));
			cuvSafeCall(cudaMemcpy(m_blocks.ptr, (void*)&blocks.front(), SPARSE_DIA_BLOCK_SIZE_LEN*blocks.size()*sizeof(int),cudaMemcpyHostToDevice));
			m_blocks.len = blocks.size();
		}
	template<class V,class I>
		dev_block_descriptor<V,I>::~dev_block_descriptor(){
			cuvSafeCall(cudaFree(m_blocks.ptr));
		}

	namespace densedense_to_dia_impl{
		template<class value_type, class index_type>
			void densedense_to_dia(
					dev_dia_matrix<value_type,index_type>& dst,
					const dev_block_descriptor<value_type,index_type>& bd,
					const dev_dense_matrix<value_type,cuv::column_major,index_type>& A,
					const dev_dense_matrix<value_type,cuv::column_major,index_type>& B){
				dim3 block(SPARSE_DIA_BLOCK_SIZE, SPARSE_DIA_BLOCK_SIZE);
				dim3 grid(bd.blocks().len);
				cout << "dMultiplyAdd: block:" << block.x << ", "<<block.y<<"; grid: "<<grid.x<<endl;
				value_type* c   = dst.vec()->ptr();
				dense2dia_mm<value_type><<<grid,block>>>(c, A.ptr(), B.ptr(), A.w(), B.w(), bd.blocks().ptr, dst.stride());
				cout << "MatrixInfo: Need to calculate " << bd.blocks().len << " of " << dst.n()/(SPARSE_DIA_BLOCK_SIZE*SPARSE_DIA_BLOCK_SIZE) <<" blocks"<<endl;
				cuvSafeCall(cudaThreadSynchronize());
			}
	}

	template<class __dia_type, class __bd_type, class __dense_type >
	void densedense_to_dia(
		   __dia_type&dst,
		   const __bd_type&bd,
		   const __dense_type&A,
		   const __dense_type&B){
		densedense_to_dia_impl::densedense_to_dia(dst,bd,A,B);
	}

	/*
	 * Instantiations
	 */
#define INST_DD2DIA(V) \
	template void densedense_to_dia(                                \
			dev_dia_matrix<V>& ,                                  \
			const dev_block_descriptor<V>& ,                      \
			const dev_dense_matrix<V,cuv::column_major>& ,        \
			const dev_dense_matrix<V,cuv::column_major>& );       \

INST_DD2DIA(float);


template class dev_block_descriptor<float>;



} // cuv







