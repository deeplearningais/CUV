#include <iostream>
#include "../../basics/dense_matrix.hpp"
#include "kernels.hpp"
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
	template <class __matrix_type>
	void 
	pairwise_distance(__matrix_type& result, const __matrix_type& A, const __matrix_type& B){
		/*cuvAssert(A.w() == B.w());*/
		/*cuvAssert(A.h() == result.h());*/
		/*cuvAssert(B.h() == result.w());*/

		const int BLOCK_DIM = 32;
		dim3 threads(BLOCK_DIM, BLOCK_DIM);
		dim3 grid(B.h() / threads.x, A.h() / threads.y);

		/*cuvAssert(B.w()%threads.x == 0);*/
		/*cuvAssert(A.h()%threads.y == 0);*/
		/*cuvAssert(B.h()%threads.x == 0);*/
		/*cuvAssert(A.w()%threads.y == 0);*/

		cuvAssert(grid.x > 0);
		cuvAssert(grid.y > 0);
		pairwise_distance_kernel<BLOCK_DIM><<< grid,threads >>>(result.ptr(),A.ptr(),B.ptr(),A.w(),B.h());
		cudaThreadSynchronize();
		checkCudaError("kernel sqDiff invocation");
	}
typedef dense_matrix<float, row_major, dev_memory_space, unsigned int> dm_rmf;
template void pairwise_distance<dm_rmf>(dm_rmf&, const dm_rmf &, const dm_rmf&);

}}}
