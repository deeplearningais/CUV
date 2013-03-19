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


#ifndef __THEANO_CONVOLUTIONS_HPP__
#define __THEANO_CONVOLUTIONS_HPP__

#include <cuv.hpp>

namespace cuv{

namespace theano_conv{

//void printdiff(timeval& start, timeval& end, long int nIter);
void initcuda();
void finalize_cuda();
void convolve_2d(cuv::tensor<float,cuv::dev_memory_space>& out, const cuv::tensor<float,cuv::dev_memory_space>& images, const cuv::tensor<float,cuv::dev_memory_space>& kern, const std::string& mode, int version=-1);

void d_convolve_d_images(cuv::tensor<float,cuv::dev_memory_space>& images, const cuv::tensor<float,cuv::dev_memory_space>& out, const cuv::tensor<float,cuv::dev_memory_space>& kern, const std::string& mode);

void d_convolve_d_kern(cuv::tensor<float,cuv::dev_memory_space>& kern_, const cuv::tensor<float,cuv::dev_memory_space>& images, const cuv::tensor<float,cuv::dev_memory_space>& out, const std::string& mode);

}
}
#endif /* __THEANO_CONVOLUTIONS_HPP__ */
