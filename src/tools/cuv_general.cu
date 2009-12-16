#include <string>
#include <stdexcept>
#include <iostream>
#include <cuda.h>
#include <cutil_inline.h>

#include "cuv_general.hpp"
#include "exception_helper.hpp"

namespace cuv{
	using namespace std;
	void cuvAssertFailed(const char *msg){
			/*cout << "cuvAssert failed: " << msg <<endl;*/
			/*abort();*/
		ExceptionTracer et;
			throw std::runtime_error(std::string(msg));
	}
	void checkCudaError(const char *msg)
	{
		cudaError_t err = cudaGetLastError();
		if( cudaSuccess != err) 
		{
			/*cout << "checkCudaError: " << msg << ": " << cudaGetErrorString(err) <<endl;*/
			/*abort();*/
			ExceptionTracer et;
			throw std::runtime_error(std::string(msg) + cudaGetErrorString(err) );
		}                         
	}
	void initCUDA(int dev){
		cutilSafeCall(cudaSetDevice(dev));
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		bool canHostmap = prop.canMapHostMemory;
		if(canHostmap){
			cutilSafeCall(cudaSetDeviceFlags(cudaDeviceMapHost));
		}
	}
	void exitCUDA(){
		cudaThreadExit();
	}

}
