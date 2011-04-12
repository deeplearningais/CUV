#include <cuv/basics/tensor.hpp>
#include <cuv/basics/cuda_array.hpp>

texture<float,         2, cudaReadModeElementType> cuda_array_tex_float; 
texture<unsigned char, 2, cudaReadModeElementType> cuda_array_tex_uchar; 

template<class T> struct texref{ };
template<> struct texref<float>{
	typedef texture<float, 2, cudaReadModeElementType> type;
	static type& get(){ return cuda_array_tex_float; }; 
};
template<> struct texref<unsigned char>{
	typedef texture<unsigned char, 2, cudaReadModeElementType> type;
	static type& get(){ return cuda_array_tex_uchar; }; 
};

namespace cuv{

template<class T> struct single_to_4{};
template<>        struct single_to_4<float>        {typedef float4 type;};
template<>        struct single_to_4<unsigned char>{typedef uchar4 type;};

template<class V,class S, class I>
void cuda_array<V,S,I>::alloc(){
	cuvAssert(m_ptr==NULL);
	typedef typename single_to_4<V>::type V4;
	cudaChannelFormatDesc channelDesc  = cudaCreateChannelDesc<V>();
	cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc<V4>();
	switch(m_dim){
		case 1:
			cudaMallocArray(&m_ptr, &channelDesc, m_width, m_height);
			break;
		case 4:
			cudaMallocArray(&m_ptr, &channelDesc4, m_width, m_height);
			break;
		default:
			cuvAssert(false);
	}
	checkCudaError("cudaMallocArray");
}
template<class V,class S, class I>
void cuda_array<V,S,I>::dealloc(){
	if(m_ptr!=NULL){
		cudaFreeArray(m_ptr);
		m_ptr = NULL;
	}
}

/*
 *template<class V,class S, class I>
 *void cuda_array<V,S,I>::bind()const{
 *    cuvAssert(m_ptr!=NULL);
 *    typedef typename texref<V>::type textype;
 *    textype& tex = texref<V>::get();
 *    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<V>();
 *    tex.normalized = false;
 *    tex.filterMode = cudaFilterModePoint;
 *    tex.addressMode[0] = cudaAddressModeClamp;
 *    tex.addressMode[1] = cudaAddressModeClamp;
 *    cudaBindTextureToArray(tex, m_ptr, channelDesc);
 *    checkCudaError("cudaBindTextureToArray");
 *}
 */

/*
 *template<class V,class S, class I>
 *void cuda_array<V,S,I>::unbind()const{
 *    cuvAssert(m_ptr!=NULL);
 *    typedef typename texref<V>::type textype;
 *    textype& tex = texref<V>::get();
 *    cudaUnbindTexture(tex);
 *    checkCudaError("cudaUnbindTexture");
 *}
 */


#define CA cuda_array<V,S,I>
template<class V,class S, class I>
void cuda_array<V,S,I>::assign(const tensor<V, host_memory_space, row_major>& src){
        cuvAssert(src.shape().size()==2);
	cuvAssert(src.ptr()!=NULL);
	cuvAssert(src.shape()[1]/m_dim == m_width);
	cuvAssert(src.shape()[0]       == m_height);
	cudaMemcpyToArray(ptr(), 0, 0, src.ptr(), src.memsize(), cudaMemcpyHostToDevice);
	checkCudaError("cudaMemcpyToArray");
}
template<class V,class S, class I>
void cuda_array<V,S,I>::assign(const tensor<V,dev_memory_space,row_major>& src){
        cuvAssert(src.shape().size()==2);
	cuvAssert(src.ptr()!=NULL);
	cuvAssert(src.shape()[1]/m_dim  == m_width);
	cuvAssert(src.shape()[0]        == m_height);
	cudaMemcpyToArray(ptr(), 0, 0, src.ptr(), src.memsize(), cudaMemcpyDeviceToDevice);
	checkCudaError("cudaMemcpyToArray");
}


template<class V,class S, class I>
V
cuda_array<V,S,I>::operator()(const I& i, const I& j)const{
	cuvAssert(false); // only works with (broken) bind active!
	return 0;
}

// explicit instantiations
#define INST(V,M,I) \
	template void cuda_array<V,M,I>::alloc();   \
	template void cuda_array<V,M,I>::dealloc();   \
	template void cuda_array<V,M,I>::assign(const tensor<V,host_memory_space,row_major>&);   \
	template void cuda_array<V,M,I>::assign(const tensor<V,dev_memory_space,row_major>&);    \
	template V cuda_array<V,M,I>::operator()(const I&, const I&)const;   

INST(float,dev_memory_space,unsigned int);
INST(unsigned char,dev_memory_space,unsigned int);

}
