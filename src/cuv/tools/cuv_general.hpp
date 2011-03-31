//*LB*
// Copyright (c) 2010, University of Bonn, Institute for Computer Science VI
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of the University of Bonn 
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*LE*






#ifndef __CUV_GENERAL_HPP__
#define __CUV_GENERAL_HPP__

#ifndef CUDA_TEST_DEVICE
#  define CUDA_TEST_DEVICE 0
#endif

#define MAX_GRID_SIZE 65535

// use this macro to make sure no error occurs when cuda functions are called
#ifdef NDEBUG
#  define cuvSafeCall(X)  \
      if(strcmp(#X,"cudaThreadSynchronize()")!=0){ X; cuv::checkCudaError(#X); } 
#else
#  define cuvSafeCall(X) X; cuv::checkCudaError(#X); 
#endif

// use this macro to ensure that a condition is true.
//  in contrast to assert(), this will throw a runtime_exception, 
//  which can be translated to python.
//  Additionally, when using Linux, you get a full stack trace printed
#define cuvAssert(X)  \
  if(!(X)){ cuv::cuvAssertFailed(#X); } 

#define DBG(X) #X <<":"<<(X)<<"  "

namespace cuv{
	/// Parent struct for host and device memory tags
	struct memory_space{};
	/// Tag for host memory
	struct host_memory_space : public memory_space {};
	/// Tag for device memory
	struct dev_memory_space  : public memory_space {};

	/// fail with an error message, a stack trace and a runtime_exception (the nicest failures you've seen ^^!)
	void cuvAssertFailed(const char *msg);
	
	/// check whether cuda thinks there was an error and fail with msg, if this is the case
	void checkCudaError(const char *msg);

	/// initialize cuda to work on dev X
	void initCUDA(int dev=0);

	/// quit cuda
	void exitCUDA();

	/// synch threads from plain (non-cuda) C++
	void safeThreadSync();
}

#endif /* __CUV_GENERAL_HPP__ */
