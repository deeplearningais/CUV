
#ifndef __CUV_GENERAL_HPP__
#define __CUV_GENERAL_HPP__

#define cuvSafeCall(X)  \
  if(1){ X; cuv::checkCudaError(#X); } 

#define cuvAssert(X)  \
  if(!(X)){ cuv::cuvAssertFailed(#X); } 

namespace cuv{
	struct memory_space{};
	struct host_memory_space : public memory_space {};
	struct dev_memory_space  : public memory_space {};

	void cuvAssertFailed(const char *msg);
	void checkCudaError(const char *msg);
	void initCUDA(int dev=0);
	void exitCUDA();
}

#endif /* __CUV_GENERAL_HPP__ */
