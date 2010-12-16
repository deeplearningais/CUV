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





/**
 *
 *
 * @mainpage
 *
 * @section summary  Summary
 *
 * CUV is a C++ template and Python library which makes it easy to use NVIDIA(tm) CUDA.
 *
 * @section features  Features
 *
 * Supported Platforms:
 * - This library was only tested on Ubuntu Karmic. It uses only standard
 *   components and should run without major modification on any current linux
 *   system.
 *
 * Structure: 
 * - Like for example Matlab, CUV assumes that everything is a matrix or a vector.
 * - Vectors/Matrices can have an arbitrary type and can be on the host (CPU-memory) or device (GPU-memory)
 * - Matrices can be column-major or row-major
 * - The library defines many functions which may or may not apply to all possible combinations. Variations are easy to add.
 * - Conversion routines are provided for most cases
 * - For convenience, we also wrap some of the functionality provided by Alex
 *   Krizhevsky on his website (http://www.cs.utoronto.ca/~kriz/) with
 *   permission. Thanks Alex for providing your code!
 *
 * Python Integration 
 * - CUV plays well with python and numpy. That is, once you wrote your fast
 *   GPU functions in CUDA/C++, you can export them using Boost.Python. You can
 *   use Numpy for pre-processing and fancy stuff you have not yet implemented,
 *   then push the Numpy-matrix to the GPU, run your operations there, pull
 *   again to CPU and visualize using matplotlib. Great.
 *
 * Implemented Functionality
 * - Simple Linear Algebra for dense vectors and matrices (BLAS level 1,2,3)
 * - Helpful functors and abstractions
 * - Sparse matrices in DIA format and matrix-multiplication for these matrices
 * - I/O functions using boost.serialization
 * - Fast Random Number Generator
 * - Up to now, CUV was used to build dense and sparse Neural Networks and
 *   Restricted Boltzmann Machines (RBM), convolutional or locally connected.
 * 
 * Documentation
 * - Tutorials are available on 
 *   http://www.ais.uni-bonn.de/~schulz/tag/cuv
 * - The API documentation can be generated from the code or accessed on the
 *   internet: http://www.ais.uni-bonn.de/deep_learning/doc/html/index.html
 *
 * @section installation  Installation
 *
 * @subsection req  Requirements
 *
 * For C++ libs, you will need:
 * - libboost-dev >= 1.37
 * - libblas-dev
 * - libtemplate-perl -- (we might get rid of this dependency soon)
 * - NVIDIA CUDA (tm), including SDK. We support versions 2.X and 3.0.
 * - thrust library (from http://code.google.com/p/thrust/)
 * - doxygen (if you want to build the documentation yourself)
 *
 * For Python Integration, you additionally have to install
 * - pyublas -- from http://mathema.tician.de/software/pyublas
 * - python-dev 
 *
 * @subsection obtaining  Obtaining CUV
 *
 * You have two choices: 
 * - Download the tar-file from our website (http://www.ais.uni-bonn.de)
 * - Checkout our git repository 
 *   @code
 *   $ git clone git://github.com/deeplearningais/CUV.git
 *   @endcode
 *
 * @subsection instproc  Installation Procedure
 *
 * Building a debug version:
 *
 * @code
 * $ tar xzvf cuv-version-source.tar.gz
 * $ cd cuv-version-source
 * $ mkdir -p build/debug
 * $ cd build/debug
 * $ cmake -DCMAKE_BUILD_TYPE=Debug ../../
 * $ ccmake .          # adjust CUDA SDK paths to your system!
 * $ make -j
 * $ ctest             # run tests to see if it went well
 * $ make install
 * $ export PYTHONPATH=`pwd`/src/python_bindings      # only if you want python bindings
 * @endcode
 *
 * Building a release version:
 *
 * @code
 * $ tar xzvf cuv-version-source.tar.gz
 * $ cd cuv-version-source
 * $ mkdir -p build/release
 * $ cd build/release
 * $ cmake -DCMAKE_BUILD_TYPE=Release ../../
 * $ ccmake .          # adjust CUDA SDK paths to your system!
 * $ make -j
 * $ ctest             # run tests to see if it went well
 * $ export PYTHONPATH=`pwd`/src/python_bindings      # only if you want python bindings
 * @endcode
 *
 * @subsection docinst  Building the documentation
 *
 * @code
 * $ cd build/debug    # change to the build directory
 * $ make doc
 * @endcode
 *
 * @section samplecode Sample Code
 *
 * We show two brief examples. For further inspiration, please take a look at
 * the test cases implemented in the @c src/tests  directory.
 *
 * @par Pushing and pulling of memory
 *
 * C++ Code:
 * @code
 * #include <vector.hpp>
 * using namespace cuv;
 * 
 * vector<float,host_memory_space> h(256);  // reserves space in host memory
 * vector<float,dev_memory_space>  d(256);  // reserves space in device memory
 *
 * fill(h,0);                          // terse form
 * apply_0ary_functor(h,NF_FILL,0);    // more verbose
 *
 * convert(d,h);                       // push to device
 * sequence(d);                        // fill device vector with a sequence
 *
 * convert(h,d);                       // pull to host
 * for(int i=0;i<h.n();i++)
 * { 
 *   assert(d[i] == h[i]);
 * }
 * @endcode
 *
 * Python Code:
 * @code
 * import pyublas
 * import cuv_python as cp
 * import numpy as np
 *
 * h = np.zeros((1,256)).astype("float32")                 # create numpy matrix
 * d = cp.push(h)                                          # creates dev_matrix_rmf (row-major float) object
 *
 * h2 = np.zeros((1,256)).astype("float32").copy("F")      # create numpy matrix
 * d2 = cp.push(h)                                         # creates dev_matrix_cmf (column-major float) object
 *
 * cp.fill(d,1)                                            # terse form
 * cp.apply_nullary_functor(d,cp.nullary_functor.FILL,1)   # verbose form
 *
 * h = cp.pull(d)
 * assert(h.sum() == 256)
 * d.dealloc()                                             # explicitly deallocate memory (optional)
 *
 * @endcode
 *
 * @par Simple Matrix operations
 *
 * C++-Code
 * @code
 * #include <dense_matrix.hpp>
 * #include <matrix_ops.hpp>
 * using namespace cuv;
 *
 * dense_matrix<float,column_major,dev_memory_space> C(2048,2048),A(2048,2048),B(2048,2048);
 *
 * fill(C,0);         // initialize to some defined value, not strictly necessary here
 * sequence(A); 
 * sequence(B);
 *
 * apply_binary_functor(A,B,BF_MULT);  // elementwise multiplication
 * prod(C,A,B, 'n','t');               // matrix multiplication
 *
 * @endcode
 *
 * Python Code
 * @code
 * import pyublas
 * import cuv_python as cp
 * import numpy as np
 * C = cp.dev_matrix_cmf(2048,2048)   # cmf = column_major float
 * A = cp.dev_matrix_cmf(2048,2048)   
 * B = cp.dev_matrix_cmf(2048,2048)
 * cp.fill(C,0)                       # fill with some defined values, not really necessary here
 * cp.sequence(A,0)
 * cp.sequence(B,0)
 * cp.apply_binary_functor(B,A,cp.binary_functor.MULT) # elementwise multiplication
 * cp.prod(C,A,B,'n','t')                              # matrix multiplication
 * @endcode
 *
 */


/** 
 * @namespace cuv
 * @brief contains all cuv functionality
 */

namespace cuv{
}
