#include <basics/dense_matrix.hpp>
#include "cuda_array.hpp"

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

template<class V,class S, class I>
void cuda_array<V,S,I>::alloc(){
	cuvAssert(m_ptr==NULL);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<V>();
	cudaMallocArray(&m_ptr, &channelDesc, m_width, m_height);
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
void cuda_array<V,S,I>::assign(const dense_matrix<V, row_major, host_memory_space, I>& src){
	cuvAssert(src.ptr()!=NULL);
	cuvAssert(src.w()  == m_width);
	cuvAssert(src.h() == m_height);
	cudaMemcpyToArray(ptr(), 0, 0, src.ptr(), src.memsize(), cudaMemcpyHostToDevice);
	checkCudaError("cudaMemcpyToArray");
}
template<class V,class S, class I>
void cuda_array<V,S,I>::assign(const dense_matrix<V,row_major,dev_memory_space,I>& src){
	cuvAssert(src.ptr()!=NULL);
	cuvAssert(src.w()  == m_width);
	cuvAssert(src.h() == m_height);
	cudaMemcpy2DToArray(ptr(), 0, 0, src.ptr(), src.w(), src.w(), src.h(), cudaMemcpyDeviceToDevice);
	checkCudaError("cudaMemcpyToArray");
}



template<class V, class I>
__global__
void
cuda_array_get_kernel(V* output, I i, I j){
}
template<class I>
__global__
void
cuda_array_get_kernel(float* output, I i, I j){
	*output = tex2D(cuda_array_tex_float,j,i);
}
template<class I>
__global__
void
cuda_array_get_kernel(unsigned char* output, I i, I j){
	*output = tex2D(cuda_array_tex_uchar,j,i);
}

template<class V,class S, class I>
V
cuda_array<V,S,I>::operator()(const I& i, const I& j)const{
	cuvAssert(false); // only works with (broken) bind active!
	V *tmp_d, tmp_h;
	cudaMalloc(&tmp_d,sizeof(V));
	cuda_array_get_kernel<<<1,1>>>(tmp_d,i,j);
	cudaMemcpy(&tmp_h,tmp_d,sizeof(V),cudaMemcpyDeviceToHost);
	return tmp_h;
}

// explicit instantiations
#define INST(V,M,I) \
	template void cuda_array<V,M,I>::alloc();   \
	template void cuda_array<V,M,I>::dealloc();   \
	template void cuda_array<V,M,I>::assign(const dense_matrix<V,row_major,host_memory_space,I>&);   \
	template void cuda_array<V,M,I>::assign(const dense_matrix<V,row_major,dev_memory_space,I>&);    \
	template V cuda_array<V,M,I>::operator()(const I&, const I&)const;   

INST(float,dev_memory_space,unsigned int);
INST(unsigned char,dev_memory_space,unsigned int);

}
