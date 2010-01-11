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
dense2dia_mm( value_type* C, const value_type* A, const value_type* B, index_type wA, index_type hA, index_type hB, int* blockidx, int dialen)
{
	int2 blk = ((int2*) blockidx)[SPARSE_DIA_BLOCK_SIZE_LEN/2 * blockIdx.x ];

	__shared__ int dia_offsets[SPARSE_DIA_BLOCK_SIZE*2];
	int v = __mul24(SPARSE_DIA_BLOCK_SIZE,threadIdx.y) + threadIdx.x;
	if(v < SPARSE_DIA_BLOCK_SIZE*2)
		dia_offsets[v] = blockidx[SPARSE_DIA_BLOCK_SIZE_LEN * blockIdx.x + 2 + v]; // 2: the two ints read already above

	__syncthreads();

    int tx = threadIdx.x;                                                                                             
    int ty = threadIdx.y;                                                                                             
                                                                                                                      
    int aBegin = blk.y;                                                              
    int aEnd   = aBegin + hA*wA;
    int aStep  = hA * SPARSE_DIA_BLOCK_SIZE;                                                                                          
                                                                                                                      
    int bBegin = blk.x;                                                              
    int bStep  = hB * SPARSE_DIA_BLOCK_SIZE;                                                                                          
                                                                                                                      
    value_type Csub = 0;                                                                                            
                                                                                                                      
    int hatyptx = __mul24(hA,ty)+tx;                                                                                     
    int hbtyptx = __mul24(hB,ty)+tx;                                                                                     

    for (int a = aBegin, b  = bBegin;                                                                                 
             a < aEnd;                                                                                               
             a += aStep, b += bStep) {                                                                                
		/*if(tx==0 && ty == 0)*/
		/*    printf("Loop: a = %d    b = %d\n",a,b);*/
                                                                                                                      
        __shared__ value_type As[SPARSE_DIA_BLOCK_SIZE][SPARSE_DIA_BLOCK_SIZE];                                                           
        __shared__ value_type Bs[SPARSE_DIA_BLOCK_SIZE][SPARSE_DIA_BLOCK_SIZE];                                                         
                                                                                                                      
		AS(ty, tx) = A[a + hatyptx];                                                                     
		BS(ty, tx) = B[b + hbtyptx];                                                                     
                                                                                                                      
		__syncthreads();  // Synchronize to make sure the matrices are loaded                                                          
																													  
		/*printf("t(%d,%d):  a+hatyptx=%d b+hatyptx=%d\n",tx,ty,a+hatyptx, b+hatyptx);*/
		for (int k = 0; k < SPARSE_DIA_BLOCK_SIZE; ++k){
			Csub += AS(k,ty)*BS(k,tx);
		}
		__syncthreads();
    }
	/*printf("t(%d,%d) Csub: %3.1f\n", tx,ty, Csub);*/

	// diagonal in block
	int dia        = tx - ty;
	int dia_real   = blk.x - blk.y + dia;
	int dia_sparse = dia_offsets[SPARSE_DIA_BLOCK_SIZE-1+dia];
	if(dia_sparse >= 0 && blk.x+tx<hB && blk.y+ty<hA){
		int offd   =  blk.y + ty;
		int idx    = dia_sparse*dialen           // the diagonal in the final matrix
		   +         offd;                       // offset within diagonal
	   C[ idx ]  = Csub;
	}
}   

namespace cuv{
	template<class V, class I>
		dev_block_descriptor<V,I>::dev_block_descriptor(const diamat_type& mat)
		{
			m_blocks.ptr = NULL;
			thrust::host_vector<int> dia_offsets(
					thrust::device_ptr<const int>(mat.get_offsets().ptr()),
					thrust::device_ptr<const int>(mat.get_offsets().ptr()+mat.get_offsets().size()));
			std::vector<block> blocks;
			for(int i=0;i<mat.h();i+=SPARSE_DIA_BLOCK_SIZE){
				for(int j=0;j<mat.w();j+=SPARSE_DIA_BLOCK_SIZE){
					/*int upperdia = (j+SPARSE_DIA_BLOCK_SIZE-1) - i; // diagonal of upper right element of BLOCK_SIZExBLOCK_SIZE block*/
					int lowerdia = j - (i+SPARSE_DIA_BLOCK_SIZE-1); // diagonal of lower left  element of BLOCK_SIZExBLOCK_SIZE block
					block b;
					b.startx = j;
					b.starty = i;
					bool founddiag = false;
					for(int e=0; e<2*SPARSE_DIA_BLOCK_SIZE-1;e++){ // diag within block
						b.diag[e] = -1;
						for(int d=0; d< dia_offsets.size();d++){
							if(lowerdia + e == dia_offsets[d]){
								/*cout << "Found Diag: "<<i<<" "<< j<<" e=" << e <<" ld="<<lowerdia<< ", d="<<d<<" do[d]="<<dia_offsets[d]<<endl;*/
								b.diag[e] = d;
								founddiag = true;
								break; // look at next diag of block
							}
						}
					}
					if(founddiag){
						/*cout << "Found Block: " << b.startx << ", "<<b.starty<<endl;*/
						/*cout << "           : ";*/
						for(int i=0;i<2*SPARSE_DIA_BLOCK_SIZE-1;i++){
							cout << b.diag[i]<<" ";
						}
						blocks.push_back(b);
						cout <<endl;
					}
				}
			}
			size_t siz = sizeof(block) * blocks.size();
			cuvSafeCall(cudaMalloc((void**)&m_blocks.ptr, siz));
			cuvSafeCall(cudaMemcpy(m_blocks.ptr, (void*)&blocks.front(),siz,cudaMemcpyHostToDevice));
			m_blocks.len = blocks.size();
			cout << "Final Block-Set MemSize: "<< siz<<endl;
			cout << "Final Block-Set Size: "<< m_blocks.len<<endl;
			cout << "Final Block-Set  Ptr: "<< m_blocks.ptr<<endl;
			cout << "Final Block-Set Size: "<< blocks.size()<<endl;
		}
	template<class V,class I>
		dev_block_descriptor<V,I>::~dev_block_descriptor(){
			if(m_blocks.ptr)
				cuvSafeCall(cudaFree(m_blocks.ptr));
			m_blocks.ptr = NULL;
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
				cuvAssert(bd.blocks().ptr);
				cuvAssert(dst.w() == B.h());
				cuvAssert(dst.h() == A.h());
				cuvAssert(A.w()   == B.w());
				cuvAssert(A.w() % SPARSE_DIA_BLOCK_SIZE  == 0);
				cout << "dMultiplyAdd: block:" << block.x << ", "<<block.y<<"; grid: "<<grid.x<<endl;
				cout << "MatrixInfo: Need to calculate " << bd.blocks().len << " of " << dst.n()/(SPARSE_DIA_BLOCK_SIZE*SPARSE_DIA_BLOCK_SIZE) <<" blocks"<<endl;
				dense2dia_mm<value_type><<<grid,block>>>(dst.vec()->ptr(), A.ptr(), B.ptr(), A.w(), A.h(), B.h(), bd.blocks().ptr, dst.stride());
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







