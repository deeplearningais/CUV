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
void checkCudaError(const char *msg);

// use this macro to make sure no error occurs when cuda functions are called
#ifdef NDEBUG
#  define cuvSafeCall(X)  \
      if(strcmp(#X,"cudaThreadSynchronize()")!=0){ X; cuv::checkCudaError(#X); }
#else
#  define cuvSafeCall(X) X; cuv::checkCudaError(#X);
#endif

/** fail with an error message, a stack trace and a runtime_exception (the nicest failures you've seen ^^!)
 * @ingroup tools
 */
void cuvAssertFailed(const char *msg);

/**
 * @def cuvAssert
 * @ingroup tools
 * use this macro to ensure that a condition is true.
 * in contrast to assert(), this will throw a runtime_exception,
 * which can be translated to python.
 * Additionally, when using Linux, you get a full stack trace printed
 */
#define cuvAssert(X)  \
  if(__builtin_expect(!(X), 0)){ cuv::cuvAssertFailed(#X); }

void safeThreadSync();

/** quit cuda
 * @ingroup tools
 */
void exitCUDA();

/** 
 * @brief Initializes CUDA context
 *
 * @ingroup tools
 * 
 * @param dev Device to use. If passed dev<0, does not call cudaInit.
 *  Then CUDA tries to automatically find a free device.
 */
void initCUDA(int dev=0);

}

#endif
