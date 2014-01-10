#include "memory.hpp"

#include <thrust/device_ptr.h>

#include "cuda_general.hpp"
#include "meta_programming.hpp"

namespace cuv {

namespace detail {

template<class value_type>
void copy(value_type* dst, const value_type* src, size_t size, host_memory_space, host_memory_space,
        cudaStream_t stream) {
    memcpy(dst, src, size * sizeof(value_type));
}

template<class value_type, class value_type2>
void copy(value_type* dst, const value_type2* src, size_t size, host_memory_space, host_memory_space,
        cudaStream_t stream) {
    for (size_t i = 0; i < size; i++)
        dst[i] = static_cast<value_type>(src[i]);
}

template<class value_type>
void copy(value_type* dst, const value_type* src, size_t size, host_memory_space, dev_memory_space,
        cudaStream_t stream) {
    cuvSafeCall(cudaMemcpyAsync(dst, src, size * sizeof(value_type), cudaMemcpyDeviceToHost, stream));
    if (stream == 0) {
        cuvSafeCall(cudaStreamSynchronize(stream));
    }
}
template<class value_type>
void copy(value_type* dst, const value_type* src, size_t size, dev_memory_space, dev_memory_space,
        cudaStream_t stream) {
    cuvSafeCall(cudaMemcpyAsync(dst, src, size * sizeof(value_type), cudaMemcpyDeviceToDevice, stream));
    if (stream == 0) {
        cuvSafeCall(cudaStreamSynchronize(stream));
    }
}

template<class value_type, class value_type2>
void copy(value_type* dst, const value_type2* src, size_t size, host_memory_space, dev_memory_space,
        cudaStream_t stream) {
    cuvSafeCall(cudaMemcpyAsync(dst, src, size * sizeof(value_type), cudaMemcpyDeviceToHost, stream));
    if (stream == 0) {
        cuvSafeCall(cudaStreamSynchronize(stream));
    }
}

template<class value_type>
void copy(value_type* dst, const value_type* src, size_t size, dev_memory_space, host_memory_space,
        cudaStream_t stream) {
    cuvSafeCall(cudaMemcpyAsync(dst, src, size * sizeof(value_type), cudaMemcpyHostToDevice, stream));
    if (stream == 0) {
        cuvSafeCall(cudaStreamSynchronize(stream));
    }
}

template<class value_type, class value_type2>
void copy(value_type* dst, const value_type2* src, size_t size, dev_memory_space, dev_memory_space,
        cudaStream_t stream) {
    if (IsSame<value_type, value_type2>::Result::value) {
        cuvSafeCall(cudaMemcpyAsync(dst, src, size * sizeof(value_type), cudaMemcpyDeviceToDevice, stream));
        if (stream == 0) {
            cuvSafeCall(cudaStreamSynchronize(stream));
        }
    } else {
        thrust::copy(thrust::device_ptr<value_type2>(const_cast<value_type2*>(src)),
                thrust::device_ptr<value_type2>(const_cast<value_type2*>(src)) + size,
                thrust::device_ptr<value_type>(dst));
        cuvSafeCall(cudaThreadSynchronize());
    }
}

template<class value_type, class value_type2>
void copy2d(value_type* dst, const value_type2* src, size_t dpitch, size_t spitch, size_t h, size_t w,
        host_memory_space, host_memory_space, cudaStream_t stream) {
    cuvSafeCall(cudaMemcpy2DAsync(dst, dpitch * sizeof(value_type),
            src, spitch * sizeof(value_type2),
            w * sizeof(value_type), h, cudaMemcpyHostToHost, stream));
    if (stream == 0) {
        cuvSafeCall(cudaStreamSynchronize(stream));
    }
}

template<class value_type, class value_type2>
void copy2d(value_type* dst, const value_type2* src, size_t dpitch, size_t spitch, size_t h,
        size_t w, host_memory_space, dev_memory_space, cudaStream_t stream) {
    cuvSafeCall(cudaMemcpy2DAsync(dst, dpitch * sizeof(value_type), src, spitch * sizeof(value_type2),
            w * sizeof(value_type), h, cudaMemcpyDeviceToHost, stream));
    if (stream == 0) {
        cuvSafeCall(cudaStreamSynchronize(stream));
    }
}

template<class value_type, class value_type2>
void copy2d(value_type* dst, const value_type2* src, size_t dpitch, size_t spitch, size_t h,
        size_t w, dev_memory_space, host_memory_space, cudaStream_t stream) {
    cuvSafeCall(cudaMemcpy2DAsync(dst, dpitch * sizeof(value_type), src, spitch * sizeof(value_type2),
            w * sizeof(value_type), h, cudaMemcpyHostToDevice, stream));
    if (stream == 0) {
        cuvSafeCall(cudaStreamSynchronize(stream));
    }
}

template<class value_type, class value_type2>
void copy2d(value_type* dst, const value_type2* src, size_t dpitch, size_t spitch, size_t h,
        size_t w, dev_memory_space, dev_memory_space, cudaStream_t stream) {
    cuvSafeCall(cudaMemcpy2DAsync(dst, dpitch * sizeof(value_type),
            src, spitch * sizeof(value_type2),
            w * sizeof(value_type), h, cudaMemcpyDeviceToDevice, stream));
    if (stream == 0) {
        cuvSafeCall(cudaStreamSynchronize(stream));
    }
}

#define CUV_MEMORY_COPY(TYPE) \
template void copy<TYPE>(TYPE*, const TYPE*, size_t, host_memory_space, host_memory_space, cudaStream_t); \
template void copy<TYPE>(TYPE*, const TYPE*, size_t, host_memory_space, dev_memory_space, cudaStream_t); \
template void copy<TYPE>(TYPE*, const TYPE*, size_t, dev_memory_space, host_memory_space, cudaStream_t); \
template void copy<TYPE>(TYPE*, const TYPE*, size_t, dev_memory_space, dev_memory_space, cudaStream_t); \
template void copy2d<TYPE, TYPE>(TYPE*, const TYPE*, size_t, size_t, size_t, size_t, host_memory_space, host_memory_space, cudaStream_t); \
template void copy2d<TYPE, TYPE>(TYPE*, const TYPE*, size_t, size_t, size_t, size_t, host_memory_space, dev_memory_space, cudaStream_t); \
template void copy2d<TYPE, TYPE>(TYPE*, const TYPE*, size_t, size_t, size_t, size_t, dev_memory_space, host_memory_space, cudaStream_t); \
template void copy2d<TYPE, TYPE>(TYPE*, const TYPE*, size_t, size_t, size_t, size_t, dev_memory_space, dev_memory_space, cudaStream_t);

CUV_MEMORY_COPY(signed char);
CUV_MEMORY_COPY(unsigned char);
CUV_MEMORY_COPY(short);
CUV_MEMORY_COPY(unsigned short);
CUV_MEMORY_COPY(int);
CUV_MEMORY_COPY(unsigned int);
CUV_MEMORY_COPY(float);
CUV_MEMORY_COPY(double);

}

}
