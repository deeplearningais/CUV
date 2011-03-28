#include <iostream>
#include "../../basics/dense_matrix.hpp"
#include "../../tools/meta_programming.hpp"
#include "kmeans.hpp"

using cuv::column_major;
using cuv::dev_memory_space;
using cuv::host_memory_space;
using cuv::dense_matrix;
using cuv::vector;


template<int BLOCK_DIM, class T, class V>
__global__
void compute_clusters_kernel(const T* matrix, T* centers, const V* indices, const unsigned int nCols, const unsigned int nRows,
		const unsigned int num_clusters) {
	// should be called with column major matrix
	typedef typename cuv::unconst<T>::type unconst_value_type;

	extern __shared__ unsigned char ptr[]; // need this intermediate variable for nvcc :-(
	unconst_value_type* centers_shared = (unconst_value_type*) ptr;
	unconst_value_type* clustercount = (unconst_value_type*) ptr + num_clusters*blockDim.x*blockDim.y;
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
	//init cluster-center shared memory
	for (int i=0; i < num_clusters; i++){
		centers_shared[i*blockDim.x*blockDim.y+ty*blockDim.x + tx ]=0;
	}

	if (tx==0)
		for (int i=0; i < num_clusters; i++){
			clustercount[i*blockDim.y+ty]=0;
		}
	__syncthreads();

	for (unsigned int my = ty; my < nCols; my += off) {
		V index=indices[my];
		if (tx==0)
			clustercount[index*blockDim.y+ty]++;

		centers_shared[index*blockDim.x*blockDim.y + ty*blockDim.x + tx] += matrix[my * nRows + row_idx ];
	}

	__syncthreads();

	for (unsigned int offset = blockDim.y / 2; offset > 0; offset >>=1) {
		if (ty < offset) {
			for(int i=0; i<num_clusters; i++){	// loop over all clusters for final reduce
				const unsigned int v = ty+offset;
				centers_shared[i*blockDim.x*blockDim.y+ty*BLOCK_DIM+tx]+=centers_shared[i*blockDim.x*blockDim.y + v *BLOCK_DIM+tx];
				if(tx==0){
					clustercount[i*blockDim.y+ty]+=clustercount[i*blockDim.y + v];
				}
			}
		}
		__syncthreads();
	}
	
	if (ty == 0) {
			for(int i=0; i<num_clusters; i++){
				centers[row_idx + i*nRows] = centers_shared[i*blockDim.x*blockDim.y + tx] / clustercount[i*blockDim.y];
			}
	}
}

template<class V, class I>
void compute_clusters_impl(dense_matrix<V,column_major,dev_memory_space,I>& centers,
		const dense_matrix<V,column_major,dev_memory_space,I>& m,
		const cuv::vector<I,dev_memory_space,I>& indices){

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
		/*dim3 threads(BLOCK_DIM,BLOCK_DIM);*/
		dim3 threads(BLOCK_DIM,1);
		int num_clusters=cuv::maximum(indices) +1;
		/*unsigned int mem = sizeof(V) * BLOCK_DIM*(BLOCK_DIM+1) *num_clusters;//+1 to count clusters!*/
		unsigned int mem = sizeof(V) * (BLOCK_DIM+1) *num_clusters;//+1 to count clusters!

		compute_clusters_kernel<BLOCK_DIM,V,I><<<grid,threads,mem>>>(m.ptr(),centers.ptr(),indices.ptr(), m.w(),m.h(), num_clusters);
		cuvSafeCall(cudaThreadSynchronize());
	}


template<class V, class I>
void compute_clusters_impl(dense_matrix<V,column_major,host_memory_space,I>& clusters,
		const dense_matrix<V,column_major,host_memory_space,I>& data,
		const vector<I,host_memory_space,I>& indices){
	const int data_length=data.h();
	int* points_in_cluster=new int[clusters.w()];
	for (int i=0; i <clusters.w(); i++)
		points_in_cluster[i]=0;
	
	// accumulate vectors:
	V* clusters_ptr = clusters.ptr();
	const V* data_ptr = data.ptr();
	for(int i=0; i<data.w(); i++){
		const int this_cluster=indices[i];
		for(int j=0; j<data.h(); j++){
			clusters_ptr[this_cluster*data_length+j]+=data_ptr[i*data_length+j];
		}
			points_in_cluster[this_cluster]++;
	}

	// devide by number of vectors in cluster

	for (int i=0; i<clusters.w(); i++)
		for(int j=0; j<clusters.h(); j++)
			clusters_ptr[i*data_length+j]/=max(points_in_cluster[i],1);
}

namespace cuv{
namespace libs{
namespace kmeans{

template<class __data_matrix_type, class __index_vector_type>
void compute_clusters(__data_matrix_type& clusters, const __data_matrix_type& data, const __index_vector_type& indices){
	cuvAssert(data.w()==indices.size());
	cuvAssert(cuv::maximum(indices)<clusters.w()); // indices start with 0.
	compute_clusters_impl(clusters,data,indices);
	}

template void compute_clusters<dense_matrix<float, column_major, host_memory_space, unsigned int>, vector<unsigned int, host_memory_space, unsigned int> >(dense_matrix<float, column_major, host_memory_space, unsigned int>&, const dense_matrix<float, column_major, host_memory_space, unsigned int>&,const vector<unsigned int, host_memory_space, unsigned int>&);
template void compute_clusters<dense_matrix<float, column_major, dev_memory_space, unsigned int>, vector<unsigned int, dev_memory_space, unsigned int> >(dense_matrix<float, column_major, dev_memory_space, unsigned int>&, const dense_matrix<float, column_major, dev_memory_space, unsigned int>&,const vector<unsigned int, dev_memory_space, unsigned int>&);

} } }
