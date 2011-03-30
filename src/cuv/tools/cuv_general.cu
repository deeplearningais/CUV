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





#include <string>
#include <stdexcept>
#include <iostream>
#include <cuda.h>
/*#include <cutil_inline.h>*/

#include "cuv_general.hpp"
#include "exception_helper.hpp"

namespace cuv{
	using namespace std;
	void cuvAssertFailed(const char *msg){
			/*cout << "cuvAssert failed: " << msg <<endl;*/
			/*abort();*/
		/*ExceptionTracer et;*/
			throw std::runtime_error(std::string(msg));
	}
	void checkCudaError(const char *msg)
	{
		cudaError_t err = cudaGetLastError();
		if( cudaSuccess != err) 
		{
			/*cout << "checkCudaError: " << msg << ": " << cudaGetErrorString(err) <<endl;*/
			/*abort();*/
			/*ExceptionTracer et;*/
			throw std::runtime_error(std::string(msg) + cudaGetErrorString(err) );
		}                         
	}
	void initCUDA(int dev){
		cuvSafeCall(cudaSetDevice(dev));
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		bool canHostmap = prop.canMapHostMemory;
		if(canHostmap){
			cuvSafeCall(cudaSetDeviceFlags(cudaDeviceMapHost));
		}
	}
	void exitCUDA(){
		cudaThreadExit();
	}

	void safeThreadSync(){
		cudaThreadSynchronize();
		checkCudaError("Save Thread Sync");
	}

}
