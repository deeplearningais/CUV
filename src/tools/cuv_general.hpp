
#ifndef __CUV_GENERAL_HPP__
#define __CUV_GENERAL_HPP__

#define cuvSafeCall(X)  \
  if(1){ X; checkCudaError(#X); } 

namespace cuv{
	void checkCudaError(const char *msg);
	void initCUDA(int dev=0);
	void exitCUDA();
}

#endif /* __CUV_GENERAL_HPP__ */
