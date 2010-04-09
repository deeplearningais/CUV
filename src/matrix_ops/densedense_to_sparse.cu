#include <cmath>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <dense_matrix.hpp>
#include "densedense_to_sparse.hpp"

// stuff from NVIDIA SDK
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define small_grid_thread_id(void) ((__umul24(blockDim.x, blockIdx.x) + threadIdx.x))
#define large_grid_thread_id(void) ((__umul24(blockDim.x,blockIdx.x + __umul24(blockIdx.y,gridDim.x)) + threadIdx.x))
#define large_grid_thread_num(void) ((__umul24(blockDim.x,gridDim.x + __umul24(blockDim.y,gridDim.y))))
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]

#define BLOCKS_LARGE_GRID_Y 4

using namespace std;

// multiply two dense matrices and put the result in an existing sparse DIA-formated matrix
template <bool wantFactAB, bool wantFactC, class value_type, class index_type>                                                                        
__global__                                                                                                            
void                                                                                                                  
dense2dia_mm( value_type* C, const value_type* A, const value_type* B, index_type wA, index_type hA, index_type hB, int* blockidx, int dialen, const value_type factAB, const value_type factC, const unsigned char rf)
{
	const int blockid = (blockIdx.y * gridDim.x + blockIdx.x);
	int2 blk = ((int2*) blockidx)[SPARSE_DIA_BLOCK_SIZE_LEN/2 * blockid ];

	__shared__ int dia_offsets[SPARSE_DIA_BLOCK_SIZE*2];
	int v = __mul24(SPARSE_DIA_BLOCK_SIZE,threadIdx.y) + threadIdx.x;
	if(v < SPARSE_DIA_BLOCK_SIZE*2)
		dia_offsets[v] = blockidx[SPARSE_DIA_BLOCK_SIZE_LEN * blockid + 2 + v]; // 2: the two ints read already above

	__syncthreads();
	if(dia_offsets[SPARSE_DIA_BLOCK_SIZE*2-1] > 0){
		// this is a dummy block for easy grid creation
		return;
	}

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
                                                                                                                      
        __shared__ value_type As[SPARSE_DIA_BLOCK_SIZE][SPARSE_DIA_BLOCK_SIZE];                                                           
        __shared__ value_type Bs[SPARSE_DIA_BLOCK_SIZE][SPARSE_DIA_BLOCK_SIZE];                                                         
                                                                                                                      
		// allow matrices to have dimensions which are not n*SPARSE_DIA_BLOCK_SIZE
		AS(ty, tx) = a+hatyptx < aEnd         ? A[a + hatyptx] : 0;
		BS(ty, tx) = b+hbtyptx < bBegin+hB*wA ? B[b + hbtyptx] : 0;
                                                                                                                      
		__syncthreads();  // Synchronize to make sure the matrices are loaded                                                          
																													  
		for (int k = 0; k < SPARSE_DIA_BLOCK_SIZE; ++k){
			Csub += AS(k,ty) * BS(k,tx);
		}
		__syncthreads();
    }

	// diagonal in block
	int dia        = tx - ty/rf;
	int dia_sparse = dia_offsets[SPARSE_DIA_BLOCK_SIZE-1+dia];
	if(dia_sparse >= 0 && blk.x+tx<hB && blk.y+ty<hA){
		int idx    =  dia_sparse*dialen           // the diagonal in the final matrix
		   +          blk.y + ty;                 // offset within diagonal
		if(0);
		else if(wantFactAB && wantFactC)
			C[ idx ]  = factC*C[idx] + factAB*Csub;
		else if(wantFactAB && !wantFactC)
			C[ idx ]  = factAB*Csub;
		else if(!wantFactAB && wantFactC)
			C[ idx ]  = factC*C[idx] + Csub;
		else if(!wantFactAB && !wantFactC)
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
			const int rf = mat.row_fact();
			const int num_dias_within_dia_block = (SPARSE_DIA_BLOCK_SIZE-1)/rf + SPARSE_DIA_BLOCK_SIZE;
			// indices are now shifted, there are equally many positive diagonals but the negative ones
			// are partially pushed out to the left. We make sure that 0th diagonal is always at the same position!
			const int dia_offset_storage_offset = (2*SPARSE_DIA_BLOCK_SIZE-1) - num_dias_within_dia_block;
			cout << "num_dias_within_dia_block: "<<num_dias_within_dia_block<<" dia_offset_storage_offset: "<<dia_offset_storage_offset<<endl;
			for(int i=0;i<mat.h();i+=SPARSE_DIA_BLOCK_SIZE){
				for(int j=0;j<mat.w();j+=SPARSE_DIA_BLOCK_SIZE){
					/*int upperdia = (j+SPARSE_DIA_BLOCK_SIZE-1) - i; // diagonal of upper right element of BLOCK_SIZExBLOCK_SIZE block*/
					int lowerdia = j - (i+SPARSE_DIA_BLOCK_SIZE-1)/rf; // diagonal of lower left  element of BLOCK_SIZExBLOCK_SIZE block
					bool founddiag = false;
					block b;
					for(int e=0;e<2*SPARSE_DIA_BLOCK_SIZE;e++)
						b.diag[e]=-1;
					for(int e=dia_offset_storage_offset; e<2*SPARSE_DIA_BLOCK_SIZE-1;e++){ // diag within block
						typename std::map<int,I>::const_iterator it = mat.m_dia2off.find(lowerdia+e-dia_offset_storage_offset);
						if(it != mat.m_dia2off.end()){
							b.diag[e] = it->second;
							founddiag = true;
						}
					}
					if(founddiag){
						b.startx = j;
						b.starty = i;
						blocks.push_back(b);
					}
				}
			}
			while(blocks.size() % BLOCKS_LARGE_GRID_Y != 0){
				blocks.push_back(block());
				blocks.back().diag[2*SPARSE_DIA_BLOCK_SIZE-1] = 1;  // marker for "just exit!"
			}
			size_t siz = sizeof(block) * blocks.size();
			cout << "Final Block-Set Size: "<< blocks.size()<<endl;
			cout << "Final Block-Set MemSize (Mb): "<< siz/1024/1024<<endl;
			cuvSafeCall(cudaMalloc((void**)&m_blocks.ptr, siz));
			cuvSafeCall(cudaMemcpy(m_blocks.ptr, (void*)&blocks.front(),siz,cudaMemcpyHostToDevice));
			m_blocks.len = blocks.size();
			/*cout << "Final Block-Set  Ptr: "<< m_blocks.ptr<<endl;*/
			/*cout << "Final Block-Set Size: "<< blocks.size()<<endl;*/
		}
	template<class V,class I>
		dev_block_descriptor<V,I>::~dev_block_descriptor(){
			if(m_blocks.ptr)
				cuvSafeCall(cudaFree(m_blocks.ptr));
			m_blocks.ptr = NULL;
		}

	namespace densedense_to_dia_impl{
		/*
		 *  For a given number of blocks, return a 2D grid large enough to contain them
		 *  FROM NVIDIA SDK
		 */
		dim3 make_large_grid(const unsigned int num_blocks){
			if (num_blocks <= 65535){
				return dim3(num_blocks);
			} else {
				unsigned int side = (unsigned int) ceil(sqrt((double)num_blocks));
				return dim3(side,side);
			}
		}

		dim3 make_large_grid(const unsigned int num_threads, const unsigned int blocksize){
			const unsigned int num_blocks = DIVIDE_INTO(num_threads, blocksize);
			if (num_blocks <= 65535){
				//fits in a 1D grid
				return dim3(num_blocks);
			} else {
				//2D grid is required
				const unsigned int side = (unsigned int) ceil(sqrt((double)num_blocks));
				return dim3(side,side);
			}
		}
		template<class value_type, class index_type>
			void densedense_to_dia(
					dia_matrix<value_type,host_memory_space,index_type>& dst,
					const host_block_descriptor<value_type,index_type>& bd,
					const dense_matrix<value_type,cuv::column_major,host_memory_space,index_type>& A,
					const dense_matrix<value_type,cuv::column_major,host_memory_space,index_type>& B,
					const value_type& factAB,
					const value_type& factC){
				cuvAssert(dst.w() == B.h());
				cuvAssert(dst.h() == A.h());
				cuvAssert(A.w()   == B.w());
				value_type *dst_diabase = dst.vec().ptr();
				const index_type Ah = A.h(), Aw = A.w(), Bh = B.h(), Bw = B.w(), Ch = dst.h(), Cw = dst.w();
				const int rf = dst.row_fact();
				for(int dia=0;dia<dst.num_dia();dia++, dst_diabase += dst.stride()){
						const int k = dst.get_offset(dia);  //diagonal offset

						const index_type row_start = rf*std::max((int)0,-k);
						const index_type col_start =  1*std::max((int)0, k);

						// number of elements to process
						const index_type N   = std::min(Ch - row_start, rf*(Cw - col_start));

						// data vectors
						value_type *const d_base = dst_diabase + row_start;
						const value_type *const a_base = A.ptr() + row_start;
						const value_type *const b_base = B.ptr() + col_start;
						const value_type *a = a_base;
						const value_type *b = b_base;

						// now the main loop: move along the columns of A and B
						// and update the corresponding data point on the diagonal
						// this is better than finishing one diagonal element and then move to the next,
						// since in that case, one has to move in Ah-sized steps.
						const value_type*const a_end = a_base+Aw*Ah;
						const value_type*const d_end = d_base+N;
						for(;a<a_end; a+=Ah,b+=Bh){
							value_type* d = d_base;
							while(d<d_end){
								for(int row_fact=1;row_fact<rf;row_fact++) // TODO: inefficient, needs explicit instantiation for fixed rf
									*d++  += (*a++)  *  (*b);
								*d++  += (*a++)  *  (*b++);
							}
							a-=N;   b-=N/rf;
						}
				}
			}
		template<class value_type, class index_type>
			void densedense_to_dia(
					dia_matrix<value_type,dev_memory_space,index_type>& dst,
					const dev_block_descriptor<value_type,index_type>& bd,
					const dense_matrix<value_type,cuv::column_major,dev_memory_space,index_type>& A,
					const dense_matrix<value_type,cuv::column_major,dev_memory_space,index_type>& B,
					const value_type& factAB,
					const value_type& factC
					){
				dim3 block(SPARSE_DIA_BLOCK_SIZE, SPARSE_DIA_BLOCK_SIZE);
				dim3 grid; 
				if(bd.blocks().len < 4096)
					grid = dim3(bd.blocks().len);
				else{
					static const int div = BLOCKS_LARGE_GRID_Y;
					cuvAssert( bd.blocks().len % div == 0 );
					int i = bd.blocks().len/div;
					grid = dim3(div,i);
				}

				cuvAssert(bd.blocks().ptr);
				cuvAssert(dst.w() == B.h());
				cuvAssert(dst.h() == A.h());
				cuvAssert(A.w()   == B.w());
				/*cuvAssert(A.w() % SPARSE_DIA_BLOCK_SIZE  == 0);*/
				/*cout << "dMultiplyAdd: block:" << block.x << ", "<<block.y<<"; grid: "<<grid.x<<endl;*/
#ifndef NDEBUG
				/*float theoret_speedup = (dst.n()/(SPARSE_DIA_BLOCK_SIZE*SPARSE_DIA_BLOCK_SIZE)) / (float)(bd.blocks().len);*/
				/*cout << "MatrixInfo: Need to calculate " << bd.blocks().len << " of " << dst.n()/(SPARSE_DIA_BLOCK_SIZE*SPARSE_DIA_BLOCK_SIZE) <<" blocks, theoretical speedup:"<< theoret_speedup<<endl;*/
#endif
				if(0);
				else if(factAB==1.f && factC==0.f)
					dense2dia_mm<false,false,value_type><<<grid,block>>>(dst.vec().ptr(), A.ptr(), B.ptr(), A.w(), A.h(), B.h(), bd.blocks().ptr, dst.stride(),factAB,factC,dst.row_fact());
				else if(factAB==1.f && factC!=0.f)
					dense2dia_mm<false,true,value_type><<<grid,block>>>(dst.vec().ptr(), A.ptr(), B.ptr(), A.w(), A.h(), B.h(), bd.blocks().ptr, dst.stride(),factAB,factC,dst.row_fact());
				else if(factAB!=1.f && factC==0.f)
					dense2dia_mm<true,false,value_type><<<grid,block>>>(dst.vec().ptr(), A.ptr(), B.ptr(), A.w(), A.h(), B.h(), bd.blocks().ptr, dst.stride(),factAB,factC,dst.row_fact());
				else if(factAB!=1.f && factC!=0.f)
					dense2dia_mm<true,true,value_type><<<grid,block>>>(dst.vec().ptr(), A.ptr(), B.ptr(), A.w(), A.h(), B.h(), bd.blocks().ptr, dst.stride(),factAB,factC,dst.row_fact());

				cuvSafeCall(cudaThreadSynchronize());
			}
	}

	template<class __dia_type, class __bd_type, class __dense_type >
	void densedense_to_dia(
		   __dia_type&dst,
		   const __bd_type&bd,
		   const __dense_type&A,
		   const __dense_type&B,
		   const typename __dia_type::value_type& factAB,
		   const typename __dia_type::value_type& factC
		   ){
		densedense_to_dia_impl::densedense_to_dia(dst,bd,A,B,factAB,factC);
	}

	/*
	 * Instantiations
	 */
#define INST_DD2DIA(V) \
	template void densedense_to_dia(                                \
			dia_matrix<V,dev_memory_space>& ,                                  \
			const dev_block_descriptor<V>& ,                      \
			const dense_matrix<V,cuv::column_major,dev_memory_space>& ,        \
			const dense_matrix<V,cuv::column_major,dev_memory_space>&,         \
			const V&,const V&);       \
	template void densedense_to_dia(                                \
			dia_matrix<V,host_memory_space>& ,                                  \
			const host_block_descriptor<V>& ,                      \
			const dense_matrix<V,cuv::column_major,host_memory_space>& ,        \
			const dense_matrix<V,cuv::column_major,host_memory_space>&,         \
			const V&,const V&);       

INST_DD2DIA(float);


template class dev_block_descriptor<float>;
template class host_block_descriptor<float>;



} // cuv







