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


#ifndef __THEANO_OPS_HPP__
#define __THEANO_OPS_HPP__

#include <cuv.hpp>

namespace cuv{
/*
 * Wrappers for theano CUDA convolution functions
 */

/** @defgroup convolution_ops_theano Convolution and pooling operations
* @{
*/

namespace theano_ops{


/**
 * initializes cuda using theano implementation 
 *
 */
//void printdiff(timeval& start, timeval& end, long int nIter);
void initcuda();
/**
 * finalizes cuda using theano implementation 
 *
 */
void finalize_cuda();


/**
 *  shuffles the dimensions of the tensor in a specific order
 *
 *  @param dst      The result of the dimension shuffle is written here
 *  @param src      The input tensor
 *  @param new_dims The order, specifing how which dimensions to shuffle
 *  @param size     The number of dimensions of the tensor
 *
 */
void dim_shuffle2(cuv::tensor<float,cuv::dev_memory_space>& dst, const cuv::tensor<float,cuv::dev_memory_space>& src, int new_dims[], unsigned int nd);

void dim_shuffle_vec(cuv::tensor<float,cuv::dev_memory_space>& dst, const cuv::tensor<float,cuv::dev_memory_space>& src, std::vector<int> pattern);
    

template<std::size_t D>
void dim_shuffle(cuv::tensor<float,cuv::dev_memory_space>& dst, const cuv::tensor<float,cuv::dev_memory_space>& src, const cuv::extent_gen<D>& eg){
    int new_dims[D];
    for (int i = 0; i < D; ++i)
    {
        new_dims[i] = eg.ranges_[i].finish();
    }
    dim_shuffle2(dst,src, new_dims, D);
}
/**
 *  flips the 2nd and 3rd dimension of the tensor
 *
 *  @param dst      The result of the flipping is written here
 *  @param src      The input tensor on which the operation is performed
 *
 */
void flip_dim2and3(cuv::tensor<float,cuv::dev_memory_space>& dst, const cuv::tensor<float,cuv::dev_memory_space>& src);


/** @} */ //end group convolution_ops_theano
}
}
#endif /* __THEANO_OPS_HPP__ */

