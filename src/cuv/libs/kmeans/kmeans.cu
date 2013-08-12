#include <iostream>
#include <cuv/basics/tensor.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/tools/meta_programming.hpp>
#include <cuv/libs/kmeans/kmeans.hpp>

#include <thrust/device_ptr.h>
#include<thrust/functional.h>
/*#include<thrust/scan.h>*/
/*#include<thrust/copy.h>*/
#include <thrust/sort.h>
/*#include <thrust/sequence.h>*/
/*#include <thrust/gather.h>*/

using cuv::column_major;
using cuv::dev_memory_space;
using cuv::host_memory_space;
using cuv::tensor;


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

template<class V, class I, class M>
void compute_clusters_impl(tensor<V,dev_memory_space,M>& centers,
		const tensor<V,dev_memory_space,M>& m,
		const cuv::tensor<I,dev_memory_space>& indices){

		I height = cuv::IsSame<M,column_major>::Result::value ? m.shape(0) : m.shape(1);
		I  width = cuv::IsSame<M,column_major>::Result::value ? m.shape(1) : m.shape(0);

		static const int BLOCK_DIM = 16;
		const int blocks_needed = ceil((float)height/(BLOCK_DIM));
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

		compute_clusters_kernel<BLOCK_DIM,V,I><<<grid,threads,mem>>>(m.ptr(),centers.ptr(),indices.ptr(), width, height, num_clusters);
		cuvSafeCall(cudaThreadSynchronize());
	}


template<class V, class I, class M>
void compute_clusters_impl(tensor<V,host_memory_space,M>& clusters,
		const tensor<V,host_memory_space,M>& data,
		const tensor<I,host_memory_space>& indices){

	const I data_length = cuv::IsSame<M,column_major>::Result::value ? data.shape(0) : data.shape(1);
	const I  data_num   = cuv::IsSame<M,column_major>::Result::value ? data.shape(1) : data.shape(0);

	const I clusters_length = cuv::IsSame<M,column_major>::Result::value ? clusters.shape(0) : clusters.shape(1);
	const I clusters_num    = cuv::IsSame<M,column_major>::Result::value ? clusters.shape(1) : clusters.shape(0);

	unsigned int* points_in_cluster=new unsigned int[clusters_num];
	for (int i=0; i <clusters_num; i++)
		points_in_cluster[i]=0;
	
	// accumulate vectors:
	V* clusters_ptr = clusters.ptr();
	const V* data_ptr = data.ptr();
	for(int i=0; i<data_num; i++){
		const int this_cluster=indices[i];
		for(int j=0; j<data_length; j++){
			clusters_ptr[this_cluster*data_length+j]+=data_ptr[i*data_length+j];
		}
			points_in_cluster[this_cluster]++;
	}

	// divide by number of vectors in cluster
	for (int i=0; i<clusters_num; i++)
		for(int j=0; j<clusters_length; j++)
			clusters_ptr[i*data_length+j]/=max(points_in_cluster[i],1);
	delete[] points_in_cluster;
}





namespace cuv{
namespace libs{
namespace kmeans{

namespace impl{

	template<class V, class L>
	thrust::device_ptr<V>
	thrust_ptr(const cuv::tensor<V,dev_memory_space,L>& m){ return thrust::device_ptr<V>(const_cast<V*>(m.ptr())); }

	template<class V, class L>
	V*
	thrust_ptr(const cuv::tensor<V,host_memory_space,L>& m){ return const_cast<V*>(m.ptr()); }

	template<class value_type, class index_type, class size_type>
	__global__
	void reorder_kernel(value_type* dst, const value_type* src, index_type* index,size_type dataDim, size_type rows){
		for(unsigned int row = blockIdx.x; row<rows; row+=gridDim.x){
			const unsigned int off = threadIdx.x;
			const unsigned int index_row = index[row];
			for (unsigned int col = off; col < dataDim; col += blockDim.x)
				dst[row*dataDim+col] = src[index_row*dataDim+col ];
		}
	}
	template<class V, class I, class M>
	void sort_by_index(tensor<V,host_memory_space,M>& sorted,
			tensor<I,host_memory_space>& indices,
			const tensor<V,host_memory_space,M>& data){

		const I data_length = cuv::IsSame<M,column_major>::Result::value ? data.shape(0) : data.shape(1);
		const I  data_num   = cuv::IsSame<M,column_major>::Result::value ? data.shape(1) : data.shape(0);

		cuv::tensor<I,host_memory_space> seq(indices.shape());
		thrust::sequence(thrust_ptr(seq),thrust_ptr(seq)+data_num);
		thrust::sort_by_key(thrust_ptr(indices),thrust_ptr(indices)+data_num,thrust_ptr(seq));
		V* sorted_ptr = sorted.ptr();
		for(I i=0;i<data.shape(0); i++){
			const V* src_ptr = data.ptr()   + data_length * seq[i];
			V* dst_ptr = sorted.ptr() + data_length * i;
			for(I j=0;j<data_length; j++)
				dst_ptr[j] = src_ptr[j];
		}
	}
	template<class V, class I, class M>
	void sort_by_index(tensor<V,dev_memory_space,M>& sorted,
			tensor<I,dev_memory_space>& indices,
			const tensor<V,dev_memory_space,M>& data){
		tensor<I,dev_memory_space> seq(indices.shape());

		const I data_length = cuv::IsSame<M,column_major>::Result::value ? data.shape(0) : data.shape(1);
		const I  data_num   = cuv::IsSame<M,column_major>::Result::value ? data.shape(1) : data.shape(0);

		thrust::sequence(thrust_ptr(seq),thrust_ptr(seq)+data.shape(0));
		thrust::sort_by_key(thrust_ptr(indices),thrust_ptr(indices)+data.shape(0),thrust_ptr(seq));

		unsigned int num_threads = min(256,data_length);
		num_threads = 32 * ((num_threads+32-1)/32); // make sure it can be divided by 32
		unsigned int num_blocks  = min(65536-1,data_num);

		reorder_kernel<<<num_blocks,num_threads>>>(sorted.ptr(),data.ptr(),seq.ptr(),data_length, data_num); 
		cuvSafeCall(cudaThreadSynchronize());
		/*thrust::sort(thrust_ptr(indices),thrust_ptr(indices)+indices.size());*/ // thrust sorts BOTH, indices AND seq.
	}
} // namespace impl

template<class __data_value_type, class __memory_space_type, class __memory_layout_type>
void compute_clusters(cuv::tensor<__data_value_type, __memory_space_type, __memory_layout_type>& clusters,
		const cuv::tensor<__data_value_type, __memory_space_type, __memory_layout_type>& data,
		const cuv::tensor<typename cuv::tensor<__data_value_type, __memory_space_type, __memory_layout_type>::index_type,__memory_space_type>& indices){
        cuvAssert(clusters.ndim()==2);
        cuvAssert(data.ndim()==2);
        cuvAssert(indices.ndim()==1);
	if(IsSame<__memory_layout_type,column_major>::Result::value){
		cuvAssert(data.shape(1)==indices.size());
		cuvAssert(cuv::maximum(indices)<clusters.shape(1)); // indices start with 0.
	}else{
		cuvAssert(data.shape(0)==indices.size());
		cuvAssert(cuv::maximum(indices)<clusters.shape(0)); // indices start with 0.
	}
	compute_clusters_impl(clusters,data,indices);
	}

template<class __data_value_type, class __memory_space_type, class __memory_layout_type>
void sort_by_index(cuv::tensor<__data_value_type, __memory_space_type, __memory_layout_type>& sorted,
		cuv::tensor<typename cuv::tensor<__data_value_type, __memory_space_type, __memory_layout_type>::index_type,__memory_space_type>& indices,
		const cuv::tensor<__data_value_type, __memory_space_type, __memory_layout_type>& data){
        cuvAssert(sorted.ndim()==2);
        cuvAssert(data.ndim()==2);
        cuvAssert(indices.ndim()==1);

	if(IsSame<__memory_layout_type,column_major>::Result::value){
		cuvAssert(data.shape(1)==indices.size());
#ifndef NDEBUG
		cuvAssert(cuv::maximum(indices)<sorted.shape(1)); // indices start with 0.
#endif
	}else{
		cuvAssert(data.shape(0)==indices.size());
#ifndef NDEBUG
		cuvAssert(cuv::maximum(indices)<sorted.shape(0)); // indices start with 0.
#endif
	}
	impl::sort_by_index(sorted,indices,data);
	}

typedef tensor<float, host_memory_space>::index_type __standard_index_type;
template void compute_clusters<float,host_memory_space,row_major>(tensor<float, host_memory_space>&, const tensor<float, host_memory_space>&,const tensor<__standard_index_type, host_memory_space>&);
template void compute_clusters<float, dev_memory_space,row_major>(tensor<float,  dev_memory_space>&, const tensor<float,  dev_memory_space>&,const tensor<__standard_index_type,  dev_memory_space>&);

template void compute_clusters<float,host_memory_space,column_major>(tensor<float, host_memory_space, column_major>&, const tensor<float, host_memory_space, column_major>&,const tensor<__standard_index_type, host_memory_space>&);
template void compute_clusters<float, dev_memory_space,column_major>(tensor<float,  dev_memory_space, column_major>&, const tensor<float,  dev_memory_space, column_major>&,const tensor<__standard_index_type,  dev_memory_space>&);


template void sort_by_index<float,host_memory_space,row_major>(tensor<float, host_memory_space>&,tensor<__standard_index_type, host_memory_space>&, const tensor<float, host_memory_space>&);
template void sort_by_index<float, dev_memory_space,row_major>(tensor<float,  dev_memory_space>&,tensor<__standard_index_type,  dev_memory_space>&, const tensor<float,  dev_memory_space>&);

template void sort_by_index<float,host_memory_space,column_major>(tensor<float, host_memory_space, column_major>&,tensor<__standard_index_type, host_memory_space>&, const tensor<float, host_memory_space, column_major>&);
template void sort_by_index<float, dev_memory_space,column_major>(tensor<float,  dev_memory_space, column_major>&,tensor<__standard_index_type,  dev_memory_space>&, const tensor<float,  dev_memory_space, column_major>&);

} } }
