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
