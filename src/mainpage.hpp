//*LB*
// Copyright (c) 2010, Hannes Schulz, Andreas Mueller, Dominik Scherer
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




/**
 * @mainpage
 *
 * CUV is a C++ template and Python library which makes it easy to use NVIDIA(tm) CUDA.
 *
 * Features:
 * - Like for example Matlab, CUV assumes that everything is a matrix or a vector.
 * - Vectors/Matrices can have an arbitrary type and can be on the host (CPU-memory) or device (GPU-memory)
 * - Matrices can be column-major or row-major
 * - The library defines many functions which may or may not apply to all possible combinations. Variations are easy to add.
 * - Conversion routines are provided for most cases
 * - CUV plays well with python and numpy. That is, once you wrote your fast
 *   GPU functions in CUDA/C++, you can export them using Boost.Python. You can
 *   use Numpy for pre-processing and fancy stuff you have not yet implemented,
 *   then push the Numpy-matrix to the GPU, run your operations there, pull
 *   again to CPU and visualize using matplotlib. Great.
 *
 */


/** 
 * @namespace cuv
 * @brief contains all cuv functionality
 */

namespace cuv{
}
