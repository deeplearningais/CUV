#include <iostream>
#include "../../basics/dense_matrix.hpp"
#include "../../tools/meta_programming.hpp"
#include "kmeans.hpp"

using cuv::column_major;
using cuv::dev_memory_space;
using cuv::host_memory_space;
using cuv::dense_matrix;
using cuv::vector;

/*template<class V1, class V2>*/
/*__global__*/
/*void compute_clusters_kernel(V1* clusters, const int clusters_h, const int clusters_w, const V1* data, const int data_w, const int data_h, const V2* indices){*/
	/*// threadIdx.x is entry in data vector*/
	/*// blockIdx.x */
	/*__shared__ int cluster_index;*/

/*}*/

template<int BLOCK_SIZE, class T, class V>
__global__
void compute_clusters_kernel(const T* matrix, T* vector, V* indices, const unsigned int nCols, const unsigned int nRows,
		const unsigned int num_clusters) {

	typedef typename cuv::unconst<T>::type unconst_value_type;

	extern __shared__ unsigned char ptr[]; // need this intermediate variable for nvcc :-(
	unconst_value_type* values = (unconst_value_type*) ptr;
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

	unconst_value_type sum = 0;

	for (unsigned int my = ty; my < nCols; my += off) {
		T f = matrix[my * nRows + row_idx ];
		/*rf.rv(sum,arg_index,f,my);*/
		//sum=rf(sum,f);
	}

	values[ty*BLOCK_SIZE+tx] = sum;
	/*if(functor_traits::returns_index)*/
		/*indices[ty*BLOCK_SIZE+tx] = arg_index;*/

	__syncthreads();

	for (unsigned int offset = blockDim.y / 2; offset > 0; offset >>=1) {
		if (ty < offset) {
			const unsigned int v = ty+offset;
			/*rf.rr(*/
					  /*values [ty*BLOCK_SIZE+tx],*/
					  /*indices[ty*BLOCK_SIZE+tx],*/
					  /*values [v *BLOCK_SIZE+tx],*/
					  /*indices[v *BLOCK_SIZE+tx]);*/
		}
		__syncthreads();
	}
	
	if (ty == 0) {
		/*if (functor_traits::returns_index)*/
			/*vector[row_idx] = indices[tx];*/
		/*else*/
				vector[row_idx] = values[tx];
	}
}

template<class V, class I>
void compute_clusters_impl(dense_matrix<V,column_major,dev_memory_space,I>& v,
		const dense_matrix<V,column_major,dev_memory_space,I>& m,
		const cuv::vector<I,dev_memory_space,I>& indices){

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

		const int num_clusters=v.w();
		std::cout <<" num clusters: " << num_clusters << std::endl;
		unsigned int mem = sizeof(V) * BLOCK_DIM_X*BLOCK_DIM_Y *num_clusters;

		compute_clusters_kernel<BLOCK_SIZE,V,I><<<grid,threads,mem>>>(m.ptr(),v.ptr(),indices.ptr(), m.w(),m.h(), num_clusters);
		cuvSafeCall(cudaThreadSynchronize());
	}


template<class V, class I>
void compute_clusters_impl(dense_matrix<V,column_major,host_memory_space,I>& clusters,
		const dense_matrix<V,column_major,host_memory_space,I>& data,
		const vector<I,host_memory_space,I>& indices){
	std::cout << "host implementation" <<std::endl;
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
	cuvAssert(cuv::maximum(indices)<=clusters.w());
	compute_clusters_impl(clusters,data,indices);
	}

template void compute_clusters<dense_matrix<float, column_major, host_memory_space, unsigned int>, vector<unsigned int, host_memory_space, unsigned int> >(dense_matrix<float, column_major, host_memory_space, unsigned int>&, const dense_matrix<float, column_major, host_memory_space, unsigned int>&,const vector<unsigned int, host_memory_space, unsigned int>&);

} } }
