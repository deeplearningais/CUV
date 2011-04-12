#include <iostream>
#include <cuv/basics/tensor.hpp>
#include <cuv/libs/kernels/kernels.hpp>
#define V(X) #X <<": "<<(X) <<"   "

using namespace std;

////////////////////////////////////////////////////////////////////////////////
//! for every row in A, every column in B, calculate sum of squared differences
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////

template <int BLOCK_DIM, class __value_type, class __distance_type>
__global__ 
void
pairwise_distance_kernel( __distance_type* C,const  __value_type* A, const __value_type* B, int wA, int hB)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = __mul24(__mul24(wA , BLOCK_DIM) , by);
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_DIM;

    int bBegin = __mul24(__mul24(wA , BLOCK_DIM) , bx);
    int bStep  = BLOCK_DIM;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    __distance_type Csub = 0;


	int waty = __mul24(wA,ty)+tx;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b  = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        __shared__ __distance_type AS[BLOCK_DIM][BLOCK_DIM];
        __shared__ __distance_type BS[BLOCK_DIM][BLOCK_DIM+1];

        AS[ty][tx] = (__distance_type)A[a + waty];
		BS[ty][tx] = (__distance_type)B[b + waty];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // compute squared difference.
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_DIM; ++k){
			__distance_type f = AS[ty][k]-BS[tx][k];
            Csub += f*f;
		}

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = __mul24(__mul24(hB , BLOCK_DIM) , by) + __mul24(BLOCK_DIM , bx);
    C[c + __mul24(hB , ty) + tx] = Csub;
}

namespace cuv{
namespace libs{	
	namespace kernels{
            namespace detail{
                template <class V>
                void pairwise_distance_impl(tensor<V,dev_memory_space,row_major>& result, const tensor<V,dev_memory_space,row_major>& A, const tensor<V,dev_memory_space,row_major>& B){
		const int BLOCK_DIM = 32;
		dim3 threads(BLOCK_DIM, BLOCK_DIM);
		dim3 grid(B.shape()[0] / threads.x, A.shape()[0] / threads.y);

		/*cuvAssert(B.shape()[1]%threads.x == 0);*/
		/*cuvAssert(A.shape()[0]%threads.y == 0);*/
		/*cuvAssert(B.shape()[0]%threads.x == 0);*/
		/*cuvAssert(A.shape()[1]%threads.y == 0);*/

		cuvAssert(grid.x > 0);
		cuvAssert(grid.y > 0);
		pairwise_distance_kernel<BLOCK_DIM><<< grid,threads >>>(result.ptr(),A.ptr(),B.ptr(),A.shape()[1],B.shape()[0]);
		cudaThreadSynchronize();
		checkCudaError("kernel sqDiff invocation");
                   
               }
            }
        template <class __value_type, class __memory_space_type, class __memory_layout_type>
        void pairwise_distance(tensor<__value_type,__memory_space_type,__memory_layout_type>& result, const tensor<__value_type,__memory_space_type,__memory_layout_type>& A, const tensor<__value_type,__memory_space_type,__memory_layout_type>& B){
                cuvAssert(result.ndim() ==2);
                cuvAssert(A.ndim() ==2);
                cuvAssert(B.ndim() ==2);
                cuvAssert(A.shape()[1] == B.shape()[1]);
                cuvAssert(A.shape()[0] == result.shape()[0]);
                cuvAssert(B.shape()[0] == result.shape()[1]);

                detail::pairwise_distance_impl(result,A,B);
	}
typedef tensor<float, dev_memory_space, row_major> t_rmf;

template void pairwise_distance<float, dev_memory_space, row_major>(t_rmf&, const t_rmf &, const t_rmf&);

}}}
