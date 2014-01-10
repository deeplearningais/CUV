#ifndef __CUV_GENERAL_HPP__
#define __CUV_GENERAL_HPP__

#include <cuda_runtime_api.h>
#include <stdexcept>

#ifndef CUDA_TEST_DEVICE
#  define CUDA_TEST_DEVICE 0
#endif

namespace cuv {

/** check whether cuda thinks there was an error and fail with msg, if this is the case
 * @ingroup tools
 */
static inline void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// use this macro to make sure no error occurs when cuda functions are called
#ifdef NDEBUG
#  define cuvSafeCall(X)  \
      if(strcmp(#X,"cudaThreadSynchronize()")!=0){ X; cuv::checkCudaError(#X); }
#else
#  define cuvSafeCall(X) X; cuv::checkCudaError(#X);
#endif

}

#endif
