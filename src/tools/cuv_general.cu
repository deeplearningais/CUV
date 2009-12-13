#include <string>
#include <stdexcept>
#include <iostream>
#include <cuda.h>
#include <cutil_inline.h>
#include "cuv_general.hpp"

namespace cuv{
	using namespace std;
	void checkCudaError(const char *msg)
	{
		cudaError_t err = cudaGetLastError();
		if( cudaSuccess != err) 
		{
			cout << "checkCudaError: " << msg << ": " << cudaGetErrorString(err) <<endl;
			abort();
			throw std::runtime_error(std::string(msg) + cudaGetErrorString(err) );
		}                         
	}
	void initCUDA(int dev){
		cutilSafeCall(cudaSetDevice(dev));
	}
	void exitCUDA(){
		cudaThreadExit();
	}

}
