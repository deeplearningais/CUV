#include <stdexcept>
#include <cuda.h>
#include <cutil_inline.h>

void checkCudaError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) 
	{
		throw std::runtime_error(std::string(msg) + cudaGetErrorString(err) );
	}                         
}

