#include <iostream>
#include "../../basics/dense_matrix.hpp"
#include "kmeans.hpp"

using cuv::column_major;
using cuv::dev_memory_space;
using cuv::host_memory_space;
using cuv::dense_matrix;
using cuv::vector;

template<class V1, class V2>
__global__
void compute_clusters_kernel(V1* clusters, const int clusters_h, const int clusters_w, const V1* data, const int data_w, const int data_h, const V2* indices){
	// threadIdx.x is entry in data vector
	// blockIdx.x 
	__shared__ int cluster_index;

}

template<class V, class I>
void compute_clusters_impl(dense_matrix<V,column_major,dev_memory_space,I>& clusters,
		const dense_matrix<V,column_major,dev_memory_space,I>& data,
		const cuv::vector<I,dev_memory_space,I>& indices){

	/*dim3 blocks(,,);*/
	static const int bs=16*16;
	int threads = min(bs,data.h());
	int blocks = data.w();
	std::cout << "dev implementation" <<std::endl;
	compute_clusters_kernel<<<blocks,threads>>>(clusters.ptr(), clusters.h(), clusters.w(), data.ptr(), data.w(), data.h(),indices.ptr());
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
