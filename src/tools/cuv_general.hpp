
#ifndef __CUV_GENERAL_HPP__
#define __CUV_GENERAL_HPP__

// use this macro to make sure no error occurs when cuda functions are called
#define cuvSafeCall(X)  \
  if(1){ X; cuv::checkCudaError(#X); } 

// use this macro to ensure that a condition is true.
//  in contrast to assert(), this will throw a runtime_exception, 
//  which can be translated to python.
//  Additionally, when using Linux, you get a full stack trace printed
#define cuvAssert(X)  \
  if(!(X)){ cuv::cuvAssertFailed(#X); } 

namespace cuv{
	// these are used to determine where data resides
	struct memory_space{};
	struct host_memory_space : public memory_space {};
	struct dev_memory_space  : public memory_space {};

	/// fail with an error message, a stack trace and a runtime_exception (the nicest failures you've seen ^^!)
	void cuvAssertFailed(const char *msg);
	
	/// check whether cuda thinks there was an error and fail with msg, if this is the case
	void checkCudaError(const char *msg);

	/// initialize cuda to work on dev X
	void initCUDA(int dev=0);

	/// quit cuda
	void exitCUDA();
}

#endif /* __CUV_GENERAL_HPP__ */
