//*LB*
// Copyright (c) 2009, Alexander Krizhevsky
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
//  * Neither the name of the University of Toronto 
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





/*
 * nvmatrix.cu
 *
 *  Created on: 20-Jan-2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */
#include <assert.h>
#include <cublas.h>
#include <cutil_inline.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "nvmatrix.cuh"

using namespace std;

cudaDeviceProp NVMatrix::deviceProps;
unsigned int NVMatrix::hostRndMults[NUM_RND_STREAMS];
bool NVMatrix::rndInitialized = false;

/*
 * Device random number generator pointers.
 */
unsigned int *NVMatrix::devRndMults;
unsigned long long *NVMatrix::devRndWords;

void NVMatrix::initDeviceProps() {
    int deviceCount;
    cutilSafeCall(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("There is no device supporting CUDA\n");
        exit(EXIT_FAILURE);
    }
    cutilSafeCall(cudaGetDeviceProperties(&deviceProps, 0));
}

void NVMatrix::_init(unsigned int numRows, unsigned int numCols) {
    _numRows = numRows;
    _numCols = numCols;
    _numElements = numRows * numCols;
    _ownsData = true;
    /*
     * By default, new matrices are in column-major order because that's how CUBLAS likes it.
     */
    _isTrans = true;
    _devData = NULL;
    if (_numElements > 0) {
        cublasAlloc(_numElements, sizeof(float), (void**) &_devData);
        checkCublasError("!!!! device memory allocation error\n");
    }
}

NVMatrix::NVMatrix() {
    _init(0, 0);
}

NVMatrix::NVMatrix(bool isTrans) {
    _init(0, 0);
    setTrans(isTrans);
}

NVMatrix::NVMatrix(int numRows, int numCols, bool isTrans) {
    _init(numRows, numCols);
    setTrans(isTrans);
}

NVMatrix::NVMatrix(const Matrix& like, bool copy) {
    _init(like.getNumRows(), like.getNumCols());
    _isTrans = like.isTrans();
    if (copy) {
        copyFromHost(like);
    }
}

NVMatrix::NVMatrix(const NVMatrix& like, bool copy) {
    _init(like.getNumRows(), like.getNumCols());
    _isTrans = like.isTrans();
    if(copy) {
        copyFromDevice(like);
    }
}

/*
 * Initializes NVMatrix with same dimensions as given matrix but
 * does not copy any data.
 */
NVMatrix::NVMatrix(const NVMatrix& like) {
    _init(like.getNumRows(), like.getNumCols());
    _isTrans = like.isTrans();
}

/*
 * Initializes NVMatrix with same dimensions as given matrix but
 * does not copy any data.
 */
NVMatrix::NVMatrix(const Matrix& like) {
    _init(like.getNumRows(), like.getNumCols());
    _isTrans = like.isTrans();
}

NVMatrix::NVMatrix(float* devData, int numRows, int numCols, bool isTrans) {
    _numRows = numRows;
    _numCols = numCols;
    _numElements = numRows * numCols;
    _ownsData = false;
    _devData = devData;
    _isTrans = isTrans;
}

NVMatrix::~NVMatrix() {
    if(_ownsData && _numElements > 0) {
        cublasStatus status = cublasFree(_devData);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! memory free error\n");
            exit(EXIT_FAILURE);
        }
    }
}

void NVMatrix::copyFromHost(const Matrix& hostMatrix, bool resizeDeviceMatrix) {
    if(resizeDeviceMatrix) {
        resize(hostMatrix);
    }
    copyFromHost(hostMatrix);
}

void NVMatrix::copyFromHost(const Matrix& hostMatrix) {
    assert(isSameDims(hostMatrix));
    cublasStatus status = cublasSetVector(_numElements, sizeof(float), hostMatrix.getData(), 1, _devData, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write)\n");
        exit( EXIT_FAILURE);
    }
    _isTrans = hostMatrix.isTrans();
}

void NVMatrix::copyFromDevice(const NVMatrix& devMatrix) {
    assert(isSameDims(devMatrix));
    cublasScopy(_numElements,devMatrix._devData, 1, _devData,1);
    checkCublasError("cublasScopy failed");
    _isTrans = devMatrix.isTrans();
}

void NVMatrix::copyFromDevice(const NVMatrix& devMatrix, bool resizeTarget) {
    if (resizeTarget) {
        resize(devMatrix);
    }
    copyFromDevice(devMatrix);
}

void NVMatrix::copyToHost(Matrix& hostMatrix) const {
    assert(isSameDims(hostMatrix));
    cublasStatus status = cublasGetVector(_numElements, sizeof(float), _devData, 1, hostMatrix.getData(), 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (read)\n");
        exit( EXIT_FAILURE);
    }
    hostMatrix.setTrans(_isTrans);
}

void NVMatrix::rightMult(const NVMatrix &b, float scaleAB, NVMatrix &target) const {
//    assert(&target != &b);
    assert(_numCols == b.getNumRows());
    if(&target != this) {
        target.resize(_numRows, b.getNumCols());
    }
    assert(target.getNumRows() == _numRows);
    assert(target.getNumCols() == b.getNumCols());
    if(_numRows % 64 != 0 || _numCols % 64 != 0 || b.getNumCols() % 64 != 0) {
        WARN("Matrix dimensions not divisible by 64 -- cublasSgemm performance may suffer.");
    }
    cublasSgemm(getTransChar(), b.getTransChar(), _numRows, b.getNumCols(), _numCols,
                scaleAB, _devData, getLeadingDim(), b.getDevData(), b.getLeadingDim(),
                0, target.getDevData(), getNumRows());
    checkCublasError("cublasSgemm failed");
    target._isTrans = true; //because target is now in col-major order
}

void NVMatrix::rightMult(const NVMatrix &b, float scaleAB) {
    rightMult(b, scaleAB, *this);
}

void NVMatrix::rightMult(const NVMatrix &b, NVMatrix& target) const {
    rightMult(b, 1, target);
}

/*
 * This will only work if this matrix is in column-major order! In other words,
 * if isTrans() returns true.
 */
void NVMatrix::addProduct(const NVMatrix& a, const NVMatrix &b, float scaleThis, float scaleAB) {
    assert(a.getNumCols() == b.getNumRows());
    assert(this->getNumRows() == a.getNumRows());
    assert(this->getNumCols() == b.getNumCols());
    assert(_isTrans);
    if(a.getNumRows() % 64 != 0 || a.getNumCols() % 64 != 0 || b.getNumCols() % 64 != 0) {
        WARN("Matrix dimensions not divisible by 64 -- cublasSgemm performance may suffer.");
    }
    cublasSgemm(a.getTransChar(), b.getTransChar(), a.getNumRows(), b.getNumCols(), a.getNumCols(),
                scaleAB, a.getDevData(), a.getLeadingDim(), b.getDevData(), b.getLeadingDim(),
                scaleThis, _devData, getLeadingDim());
    checkCublasError("cublasSgemm failed");
}

void NVMatrix::addProduct(const NVMatrix& a, const NVMatrix &b) {
    addProduct(a, b, 1, 1);
}

void NVMatrix::apply(NVMatrix::FUNCTIONS f, NVMatrix& target, int numBlocks, int numThreadsPerBlock) {
    target.resize(*this);
    target._isTrans = _isTrans;
    dim3  grid( numBlocks, 1, 1);
    dim3  threads( numThreadsPerBlock, 1, 1);

    if(f == NVMatrix::EXP) {
        kExp<<<grid, threads>>>(_devData, target._devData, _numElements);
    } else if (f == NVMatrix::LOGISTIC1) {
        kLogistic1<<<grid, threads>>>(_devData, target._devData, _numElements);
    } else if (f == NVMatrix::LOGISTIC2) {
        kLogistic2<<<grid, threads>>>(_devData, target._devData, _numElements);
    } else if (f == NVMatrix::SQUARE) {
        kSquare<<<grid, threads>>>(_devData, target._devData, _numElements);
    } else if (f == NVMatrix::SQRT) {
        kSqrt<<<grid, threads>>>(_devData, target._devData, _numElements);
    } else if (f == NVMatrix::ZERO) {
        kZero<<<grid, threads>>>(_devData, target._devData, _numElements);
    } else if(f == NVMatrix::RECIPROCAL) {
        kReciprocal<<<grid, threads>>>(_devData, target._devData, _numElements);
    } else if(f == NVMatrix::LOG) {
        kLog<<<grid, threads>>>(_devData, target._devData, _numElements);
    }

    cutilCheckMsg("Kernel execution failed");
}

void NVMatrix::apply(NVMatrix::FUNCTIONS f, int numBlocks, int numThreadsPerBlock) {
    apply(f, *this, numBlocks, numThreadsPerBlock);
}

/*
 * The random number generator uses the multiply with carry algorithm. I got the
 * multipliers from a site I can't find anymore.
 */
void NVMatrix::initRandom(unsigned int seed) {
    assert(!rndInitialized);
    ifstream inFile;
    inFile.open(RND_MULTIPLIERS_FILE);
    if(!inFile) {
        std::cerr << "Unable to open file " << RND_MULTIPLIERS_FILE << std::endl;
        exit(EXIT_FAILURE);
    }

    unsigned int mult;
    for (int numRead = 0; numRead < NUM_RND_STREAMS; numRead++) {
        if (!(inFile >> mult)) {
            std::cerr << "Not enough numbers in file " << RND_MULTIPLIERS_FILE << std::endl;
            exit(EXIT_FAILURE);
        }
        hostRndMults[numRead] = mult;
    }
    inFile.close();

    cutilSafeCall(cudaMalloc((void **)&devRndMults, NUM_RND_STREAMS * sizeof(unsigned int)));
    cutilSafeCall(cudaMalloc((void **)&devRndWords, NUM_RND_STREAMS * sizeof(unsigned long long)));
    cutilSafeCall(cudaMemcpy(devRndMults, hostRndMults, NUM_RND_STREAMS * sizeof(unsigned int), cudaMemcpyHostToDevice));

    kSeedRandom<<<NUM_RND_BLOCKS, NUM_RND_THREADS_PER_BLOCK>>>(devRndMults, devRndWords, seed);
    cutilCheckMsg("Kernel execution failed");
    rndInitialized = true;
}

void NVMatrix::destroyRandom() {
    assert(rndInitialized);
    cutilSafeCall(cudaFree(devRndMults));
    cutilSafeCall(cudaFree(devRndWords));
    rndInitialized = false;
}

void NVMatrix::binarizeProbs() {
    assert(rndInitialized);
    kBinarizeProbs<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(devRndMults, devRndWords, _devData,_numElements);
    cutilCheckMsg("Kernel execution failed");
}

void NVMatrix::randomizeUniform() {
    assert(rndInitialized);
    kRandomUniform<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(devRndMults, devRndWords, _devData,_numElements);
    cutilCheckMsg("Kernel execution failed");
}

void NVMatrix::randomizeGaussian() {
    randomizeGaussian(1);
}

void NVMatrix::randomizeGaussian(float stdev) {
    assert(rndInitialized);
    kRandomGaussian<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(devRndMults, devRndWords, _devData, stdev, _numElements);
    cutilCheckMsg("Kernel execution failed");
}

void NVMatrix::addGaussianNoise() {
    addGaussianNoise(1);
}

void NVMatrix::addGaussianNoise(float stdev) {
    assert(rndInitialized);
    assert(_numElements % 2 == 0);
    kAddGaussianNoise<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(devRndMults, devRndWords, _devData,stdev,_numElements);
    cutilCheckMsg("Kernel execution failed");
}

void NVMatrix::biggerThanScalar(float scalar) {
    biggerThanScalar(scalar, *this);
}
void NVMatrix::biggerThanScalar(float scalar, NVMatrix& target) {
    kBiggerThanScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(_devData,scalar,target._devData,_numElements);
    cutilCheckMsg("Kernel execution failed");
}

void NVMatrix::biggerThan(NVMatrix& m, NVMatrix& target, int numBlocks, int numThreadsPerBlock) {
    assert(isSameDims(m));
    target.resize(*this);
    for (unsigned int elementsDone = 0; elementsDone < _numElements; elementsDone += numBlocks*numThreadsPerBlock) {
        kBiggerThan<<<numBlocks, numThreadsPerBlock>>>(_devData + elementsDone,
                m._devData + elementsDone, target._devData + elementsDone,
                _numElements - elementsDone);
    }
    cutilCheckMsg("Kernel execution failed");
}

void NVMatrix::biggerThan(NVMatrix& m, int numBlocks, int numThreadsPerBlock) {
    biggerThan(m, *this, numBlocks, numThreadsPerBlock);
}

void NVMatrix::_checkBounds(int startRow, int endRow, int startCol, int endCol) const {
    assert(startRow >= 0 && startRow <= _numRows);
    assert(endRow >= 0 && endRow <= _numRows);
    assert(startCol >= 0 && startCol <= _numCols);
    assert(endCol >= 0 && endCol <= _numCols);
}

NVMatrix& NVMatrix::slice(int startRow, int endRow, int startCol, int endCol) const {
    endRow = endRow < 0 ? this->_numRows : endRow;
    endCol = endCol < 0 ? this->_numCols : endCol;
    _checkBounds(startRow, endRow, startCol, endCol);
    if (!isTrans() && ((startCol == 0 && endCol == this->_numCols) || startRow == endRow - 1)) {
        return *new NVMatrix(this->_devData + startRow * this->_numCols + startCol, endRow - startRow, endCol - startCol, false);
    } else if(isTrans() && ((startRow == 0 & endRow == this->_numRows) || startCol == endCol - 1)) {
        return *new NVMatrix(this->_devData + startCol * this->_numRows + startRow, endRow - startRow, endCol - startCol, true);
    }

    WARN("Slice: result will not be a view.");
    NVMatrix& newSlice = *new NVMatrix(endRow - startRow, endCol - startCol);
    this->copy(newSlice, startRow, endRow, startCol, endCol, 0, 0);
    return newSlice;
}

/* this will NEVER return a view, unlike Matrix_slice */
void NVMatrix::slice(int startRow, int endRow, int startCol, int endCol, NVMatrix& target) const {
    endRow = endRow < 0 ? this->_numRows : endRow;
    endCol = endCol < 0 ? this->_numCols : endCol;
    _checkBounds(startRow, endRow, startCol, endCol);
    target.resize(endRow - startRow, endCol - startCol);
    target._isTrans = _isTrans;
    this->copy(target, startRow, endRow, startCol, endCol, 0, 0);
}

NVMatrix& NVMatrix::sliceRows(int startRow, int endRow) const {
    return slice(startRow, endRow, 0, -1);
}

void NVMatrix::sliceRows(int startRow, int endRow, NVMatrix& target) const {
    slice(startRow, endRow, 0, -1, target);
}

NVMatrix& NVMatrix::sliceCols(int startCol, int endCol) const {
    return slice(0, -1, startCol, endCol);
}

void NVMatrix::sliceCols(int startCol, int endCol, NVMatrix& target) const {
    slice(0, -1, startCol, endCol, target);
}

/*
 * Guaranteed to not change the data if the number of elements doesn't change.
 * So you can use this to "reshape" a matrix.
 */
bool NVMatrix::resize(int numRows, int numCols) {
    bool reallocated = false;
    if (numRows != _numRows || numCols != _numCols) {
        assert(_ownsData);
        if (_numElements != numRows * numCols) {
            cublasStatus status = cublasFree(_devData);
            if (status != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr, "!!!! memory free error\n");
                exit(EXIT_FAILURE);
            }
            status = cublasAlloc(numCols * numRows, sizeof(float), (void**) &_devData);
            if (status != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr, "!!!! device memory allocation error\n");
                exit(EXIT_FAILURE);
            }
            reallocated = true;
        }
        _numRows = numRows;
        _numCols = numCols;
        _numElements = numRows * numCols;
    }
    return reallocated;
}

bool NVMatrix::resize(const NVMatrix& like) {
    bool r = resize(like.getNumRows(), like.getNumCols());
    _isTrans = like._isTrans;
    return r;
}

bool NVMatrix::resize(const Matrix& like) {
    bool r = resize(like.getNumRows(), like.getNumCols());
    _isTrans = like.isTrans();
    return r;
}

void NVMatrix::reshape(int numRows, int numCols) {
    assert(_numElements == numRows*numCols);
    _numRows = numRows;
    _numCols = numCols;
}

NVMatrix& NVMatrix::reshaped(int numRows, int numCols) {
    assert(_numElements == numRows*numCols);
    return *new NVMatrix(_devData, numRows, numCols, _isTrans);
}

void NVMatrix::copy(NVMatrix &dest, int srcStartRow, int srcEndRow, int srcStartCol, int srcEndCol,
                    int destStartRow, int destStartCol, int numBlocks, int numThreadsPerBlock) const {
    srcEndRow = srcEndRow < 0 ? this->_numRows : srcEndRow;
    srcEndCol = srcEndCol < 0 ? this->_numCols : srcEndCol;
    assert(destStartRow >= 0 && destStartCol >= 0); //some range-checking
    assert(srcEndRow <= _numRows && srcEndCol <= _numCols);
    assert(destStartRow + srcEndRow - srcStartRow <= dest.getNumRows());
    assert(destStartCol + srcEndCol - srcStartCol <= dest.getNumCols());

    float* srcStartPtr = getCellPtr(srcStartRow, srcStartCol);
    float* destStartPtr = dest.getCellPtr(destStartRow, destStartCol);
    const int copyWidth = !_isTrans ? srcEndCol - srcStartCol : srcEndRow - srcStartRow;
    const int copyHeight = !_isTrans ? srcEndRow - srcStartRow : srcEndCol - srcStartCol;
    const int srcJumpSize = !_isTrans ? getNumCols() : getNumRows();
    const int destJumpSize = !dest._isTrans ? dest.getNumCols() : dest.getNumRows();

    int numElements = (srcEndRow - srcStartRow) * (srcEndCol - srcStartCol);
    const int numThreads = numBlocks * numThreadsPerBlock;

    const int numCopyRows = numThreads / copyWidth;
    if(isTrans() != dest.isTrans() && !(copyWidth % COPY_BLOCK_SIZE == 0 && copyHeight % COPY_BLOCK_SIZE == 0)) {
        WARN("Matrix copy: matrices have different transposedness and copy region dimensions not divisible by 16 -- calling inefficient copy kernel.");
    }
    if(isTrans() == dest.isTrans() || !(copyWidth % COPY_BLOCK_SIZE == 0 && copyHeight % COPY_BLOCK_SIZE == 0)) {
        while (numElements > 0) {
            int numToCopy = min(numElements, numCopyRows * copyWidth);
            if(isTrans() == dest.isTrans()) {
                kCopy<<<numBlocks,numThreadsPerBlock>>>(srcStartPtr, destStartPtr, copyWidth, srcJumpSize, numToCopy);
            } else {
                kCopyToTransDestSlow<<<numBlocks,numThreadsPerBlock>>>(srcStartPtr, destStartPtr, copyWidth, srcJumpSize, destJumpSize, numToCopy);
            }
            cutilCheckMsg("Kernel execution failed");
            numElements -= numToCopy;
            srcStartPtr += isTrans() ? numCopyRows : numToCopy;
            destStartPtr += dest.isTrans() ? numCopyRows : numToCopy;
        }
    } else {
//        printf("neato copy kernel\n");
        const int numBlocksX = copyWidth / COPY_BLOCK_SIZE;
        assert(numBlocksX < NUM_BLOCKS_MAX);
        const int numBlocksY = min(copyHeight / COPY_BLOCK_SIZE, NUM_BLOCKS_MAX);
        dim3 gridSize(numBlocksX, numBlocksY, 1);
        dim3 blockSize(COPY_BLOCK_SIZE, COPY_BLOCK_SIZE, 1);
        int numRowsCopied = 0;
        while(numRowsCopied < copyHeight) {
            kCopyToTransDestFast<<<gridSize, blockSize>>>(srcStartPtr + srcJumpSize * numRowsCopied, destStartPtr + numRowsCopied, copyWidth, copyHeight, srcJumpSize, destJumpSize);
            cutilCheckMsg("Kernel execution failed");
            numRowsCopied += gridSize.y * COPY_BLOCK_SIZE;
            gridSize.y = min((copyHeight - numRowsCopied) / COPY_BLOCK_SIZE, NUM_BLOCKS_MAX);
        }
    }
}

void NVMatrix::copy(NVMatrix &dest, int srcStartRow, int srcEndRow, int srcStartCol, int srcEndCol,
                    int destStartRow, int destStartCol) const {
    //TODO: these grid/block sizes may not be good
    copy(dest, srcStartRow, srcEndRow, srcStartCol, srcEndCol, destStartRow, destStartCol,
            NUM_BLOCKS_MAX, deviceProps.maxThreadsPerBlock);
}

NVMatrix& NVMatrix::getTranspose() {
    NVMatrix* trans = new NVMatrix(_devData, _numCols, _numRows, !_isTrans);
    return *trans;
}

/*
 * Flips the ordering of the matrix from row-major to column-major and vice versa.
 * This creates temporary storage -- not a cheap operation.
 *
 * This is not equivalent to a "hard transpose". The resultant matrix still has
 * the same dimensions, its layout in memory just changes.
 */
void NVMatrix::flipTrans() {
    NVMatrix* meTrans = new NVMatrix(*this);
//    assert(_numCols % ADD_BLOCK_SIZE == 0 && _numRows % ADD_BLOCK_SIZE == 0);
    const int width = isTrans() ? _numRows : _numCols;
    const int height = isTrans() ? _numCols : _numRows;
    const int numBlocksX = DIVUP(width, ADD_BLOCK_SIZE);
    const int numBlocksY = DIVUP(height, ADD_BLOCK_SIZE);
    assert(numBlocksX < NUM_BLOCKS_MAX && numBlocksY < NUM_BLOCKS_MAX);
    dim3 gridSize(numBlocksX, numBlocksY, 1);
    dim3 blockSize(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1);
    kTranspose<<<gridSize, blockSize>>>(_devData, meTrans->_devData, width, height);
    cutilCheckMsg("Kernel execution failed");

    copyFromDevice(*meTrans);
    this->_isTrans = !this->_isTrans;
    delete meTrans;
}

void NVMatrix::squaredDiff(NVMatrix& b) {
    squaredDiff(b, *this);
}

void NVMatrix::squaredDiff(NVMatrix& b, NVMatrix& target) {
    assert(this->isSameDims(b));
    assert(&target != &b);
    target.resize(*this);
    const int width = isTrans() ? _numRows : _numCols;
    const int height = isTrans() ? _numCols : _numRows;
    if (_isTrans != b._isTrans) {
        assert(width % ADD_BLOCK_SIZE == 0 && height % ADD_BLOCK_SIZE == 0);
        const int numBlocksX = width / ADD_BLOCK_SIZE;
        assert(numBlocksX < NUM_BLOCKS_MAX);
        const int numBlocksY = std::max(1, std::min(height / ADD_BLOCK_SIZE, NUM_BLOCKS_MAX));
        dim3 gridSize(numBlocksX, numBlocksY, 1);
        dim3 blockSize(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1);
        int numRowsAdded = 0;
        float* aData = _devData, *bData = b._devData, *destData = target._devData;
//        printf("calling trans sq diff\n");
        while (numRowsAdded < height) {
            kSquaredDiffTransFast<<<gridSize, blockSize>>>(aData, bData, destData, width, height);
            cutilCheckMsg("Kernel execution failed");
            numRowsAdded += gridSize.y * ADD_BLOCK_SIZE;
            gridSize.y = std::max(1, std::min((height-numRowsAdded) / ADD_BLOCK_SIZE, NUM_BLOCKS_MAX));
            aData += numRowsAdded * width;
            bData += b._isTrans != _isTrans ? numRowsAdded : numRowsAdded * width;
            destData += numRowsAdded * width;
        }
    } else {
//        printf("calling plain sq diff\n");
        kSquaredDiff<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(_devData, b._devData, target._devData,_numElements);
        cutilCheckMsg("Kernel execution failed");
    }
}

void NVMatrix::addSum(NVMatrix& b, NVMatrix& c, float scaleThis, float scaleB, float scaleC) {
    assert(this->isSameDims(b));
    assert(this->isSameDims(c));
    const int width = isTrans() ? _numRows : _numCols;
    const int height = isTrans() ? _numCols : _numRows;
    if((_isTrans != b._isTrans || _isTrans != c._isTrans) && min(_numRows, _numCols) > 1) {
        bool checkBounds = !(width % ADD_BLOCK_SIZE == 0 && height % ADD_BLOCK_SIZE == 0);

        const int numBlocksX = DIVUP(width, ADD_BLOCK_SIZE);
        assert(numBlocksX < NUM_BLOCKS_MAX);
        const int numBlocksY = std::max(1, min(DIVUP(height, ADD_BLOCK_SIZE), NUM_BLOCKS_MAX));
        dim3 gridSize(numBlocksX, numBlocksY, 1);
        dim3 blockSize(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1);
        int numRowsAdded = 0;
        float* aData = _devData, *bData = b._devData, *cData = c._devData;
        while (numRowsAdded < height) {
            if(checkBounds) {
                kAddTrans3Fast<true><<<gridSize, blockSize>>>(aData, bData, cData,width, height - numRowsAdded, height,
                                                    scaleThis, scaleB, scaleC, b._isTrans != _isTrans, c._isTrans != _isTrans);
            } else {
                kAddTrans3Fast<false><<<gridSize, blockSize>>>(aData, bData, cData,width, height - numRowsAdded, height,
                                                        scaleThis, scaleB, scaleC, b._isTrans != _isTrans, c._isTrans != _isTrans);
            }
            cutilCheckMsg("Kernel execution failed");
            numRowsAdded += gridSize.y * ADD_BLOCK_SIZE;
            gridSize.y = std::max(1, min(DIVUP((height-numRowsAdded) , ADD_BLOCK_SIZE), NUM_BLOCKS_MAX));
            aData += numRowsAdded * width;
            bData += b._isTrans != _isTrans ? numRowsAdded : numRowsAdded * width;
            cData += c._isTrans != _isTrans ? numRowsAdded : numRowsAdded * width;
        }
    } else {
        kAdd3<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(_devData, b._devData, c._devData,
                                                                          _numElements, scaleThis, scaleB, scaleC);
        cutilCheckMsg("Kernel execution failed");
    }
}

void NVMatrix::add(NVMatrix& b, float scaleA, float scaleB, NVMatrix& target) {
    if(&target == &b && &target != this) { // because we manipulate target to be like a
        b.add(*this, scaleB, scaleA);
        return;
    }
    assert(this->isSameDims(b));
    target.resize(*this);
    if (isTrans() != b.isTrans() && min(_numRows, _numCols) > 1) {
        //call addition kernel for transposed matrices
        const int width = isTrans() ? _numRows : _numCols;
        const int height = isTrans() ? _numCols : _numRows;
        if (width % ADD_BLOCK_SIZE == 0 && height % ADD_BLOCK_SIZE == 0) {
            const int numBlocksX = width / ADD_BLOCK_SIZE;
            assert(numBlocksX < NUM_BLOCKS_MAX);
            const int numBlocksY = std::max(1, std::min(height / ADD_BLOCK_SIZE, NUM_BLOCKS_MAX));
            dim3 gridSize(numBlocksX, numBlocksY, 1);
            dim3 blockSize(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1);
            int numRowsAdded = 0;
            while (numRowsAdded < height) {
                kAddTransFast<<<gridSize, blockSize>>>(&_devData[numRowsAdded * width],
                        &b._devData[numRowsAdded], &target._devData[numRowsAdded * width],
                        width, height - numRowsAdded, height, scaleA, scaleB);
                cutilCheckMsg("Kernel execution failed");
                numRowsAdded += gridSize.y * ADD_BLOCK_SIZE;
                gridSize.y = std::max(1, std::min((height-numRowsAdded) / ADD_BLOCK_SIZE, NUM_BLOCKS_MAX));
            }
        } else {
            WARN("Add: Matrices have different transposedness and matrix dimensions not divisible by 16 -- calling inefficient matrix addition kernel.");
            kAddTransSlow<<<getDefaultNumBlocks(), getDefaultNumThreadsPerBlock()>>>(_devData, b._devData, target._devData,
                    width, height, _numElements, scaleA, scaleB);
            cutilCheckMsg("Kernel execution failed");
        }
    } else {
        if(scaleA == 1.0f) {
            cublasSaxpy(_numElements, scaleB, b._devData, 1, target._devData, 1);
            checkCublasError("cublasSaxpy failed");
        } else {
            kAdd<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(_devData, b._devData, target._devData,
                                                                            _numElements, scaleA, scaleB);
        }

    }
}

void NVMatrix::add(NVMatrix& b, float scaleB, NVMatrix& target) {
    add(b, 1, scaleB, target);
}

void NVMatrix::add(NVMatrix& b, NVMatrix& target) {
    add(b, 1, target);
}

void NVMatrix::add(NVMatrix& b, float scaleB) {
    add(b, scaleB, *this);
}

void NVMatrix::add(NVMatrix& b, float scaleA, float scaleB) {
    add(b, scaleA, scaleB, *this);
}

void NVMatrix::add(NVMatrix& b) {
    add(b, 1, *this);
}

void NVMatrix::subtract(NVMatrix& b, NVMatrix& target) {
    add(b, -1, target);
}

void NVMatrix::subtract(NVMatrix& b) {
    add(b, -1);
}

void NVMatrix::eltWiseMult(NVMatrix& b, NVMatrix& target) {
    if(&target == &b && &target != this) { // because we manipulate target to be like a
        b.eltWiseMult(*this);
        return;
    }
    assert(this->isSameDims(b));
    target.resize(*this);
    if (isTrans() != b.isTrans() && min(_numRows, _numCols) > 1) {
        //call mult kernel for transposed matrices
        const int width = isTrans() ? _numRows : _numCols;
        const int height = isTrans() ? _numCols : _numRows;
        assert(width % ADD_BLOCK_SIZE == 0 && height % ADD_BLOCK_SIZE == 0);
//        if (width % ADD_BLOCK_SIZE == 0 && height % ADD_BLOCK_SIZE == 0) {
        const int numBlocksX = width / ADD_BLOCK_SIZE;
        assert(numBlocksX < NUM_BLOCKS_MAX);
        const int numBlocksY = min(height / ADD_BLOCK_SIZE, NUM_BLOCKS_MAX);
        dim3 gridSize(numBlocksX, numBlocksY, 1);
        dim3 blockSize(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1);
        int numRowsProcessed = 0;
        while (numRowsProcessed < height) {
            kMultTransFast<<<gridSize, blockSize>>>(&_devData[numRowsProcessed * width],
                                                    &b._devData[numRowsProcessed], &target._devData[numRowsProcessed * width],
                                                    width, height - numRowsProcessed, height);
            cutilCheckMsg("Kernel execution failed");
            numRowsProcessed += gridSize.y * ADD_BLOCK_SIZE;
            gridSize.y = min((height-numRowsProcessed) / ADD_BLOCK_SIZE, NUM_BLOCKS_MAX);
        }
//        }
    } else {
        kMult<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(_devData, b._devData, target._devData,_numElements);
        cutilCheckMsg("Kernel execution failed");
    }
}

void NVMatrix::eltWiseMult(NVMatrix& b) {
    eltWiseMult(b, *this);
}

void NVMatrix::eltWiseDivide(NVMatrix& b, NVMatrix& target) {
    assert(&b != this); // doable but not necessary for me
    assert(this->isSameDims(b));
    target.resize(*this);
    if (isTrans() != b.isTrans() && min(_numRows, _numCols) > 1) {
        //call mult kernel for transposed matrices
        const int width = isTrans() ? _numRows : _numCols;
        const int height = isTrans() ? _numCols : _numRows;
        assert(width % ADD_BLOCK_SIZE == 0 && height % ADD_BLOCK_SIZE == 0);
//        if (width % ADD_BLOCK_SIZE == 0 && height % ADD_BLOCK_SIZE == 0) {
        const int numBlocksX = width / ADD_BLOCK_SIZE;
        assert(numBlocksX < NUM_BLOCKS_MAX);
        const int numBlocksY = min(height / ADD_BLOCK_SIZE, NUM_BLOCKS_MAX);
        dim3 gridSize(numBlocksX, numBlocksY, 1);
        dim3 blockSize(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1);
        int numRowsProcessed = 0;
        while (numRowsProcessed < height) {
            kDivideTransFast<<<gridSize, blockSize>>>(&_devData[numRowsProcessed * width],
                                                        &b._devData[numRowsProcessed], &target._devData[numRowsProcessed * width],
                                                        width, height - numRowsProcessed, height);
            cutilCheckMsg("Kernel execution failed");
            numRowsProcessed += gridSize.y * ADD_BLOCK_SIZE;
            gridSize.y = min((height-numRowsProcessed) / ADD_BLOCK_SIZE, NUM_BLOCKS_MAX);
        }
//        }
    } else {
        kDivide<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(_devData, b._devData, target._devData,_numElements);
        cutilCheckMsg("Kernel execution failed");
    }
}

void NVMatrix::eltWiseDivide(NVMatrix& b) {
    eltWiseDivide(b, *this);
}

void NVMatrix::tile(int timesY, int timesX, NVMatrix& target, int numBlocks, int numThreadsPerBlock) {
    assert(timesX > 0 && timesY > 0);
    target.resize(_numRows*timesY, _numCols*timesX);
    target._isTrans = _isTrans;
    if(!isTrans()) {
        kTile<<<numBlocks,numThreadsPerBlock>>>(_devData, target._devData, _numCols, _numRows, target._numCols, target._numRows);
    } else {
        kTile<<<numBlocks,numThreadsPerBlock>>>(_devData, target._devData, _numRows, _numCols, target._numRows, target._numCols);
    }
    cutilCheckMsg("Kernel execution failed");
}

void NVMatrix::addVector(NVMatrix& vec, float scaleVec, NVMatrix& target, int numBlocks, int numThreadsPerBlock) {
    if(&target == &vec && &target != this) { // because we manipulate target to be like a
        vec.add(*this, scaleVec, 1);
        return;
    }
    assert(vec.getNumRows() == 1 || vec.getNumCols() == 1);
    assert(vec.getNumRows() == _numRows || vec.getNumCols() == _numCols);
//    assert(&target != &vec);
    target.resize(*this);

//    const unsigned int numThreads = numBlocks*numThreadsPerBlock;
    const unsigned int width = _isTrans ? _numRows : _numCols;
    const unsigned int height = _isTrans ? _numCols : _numRows;
    if(vec.getNumRows() == _numRows && !isTrans() || vec.getNumCols() == _numCols && isTrans()) {
        kAddColVector<<<NUM_ADD_VECTOR_BLOCKS,NUM_ADD_VECTOR_THREADS_PER_BLOCK>>>(_devData, vec._devData, target._devData, width, height, scaleVec);
    } else {
        kAddRowVector<<<NUM_ADD_VECTOR_BLOCKS,NUM_ADD_VECTOR_THREADS_PER_BLOCK>>>(_devData, vec._devData, target._devData, width, height, scaleVec);
    }
    cutilCheckMsg("Kernel execution failed");
}

void NVMatrix::addVector(NVMatrix& vec) {
    addVector(vec, 1, *this);
}

void NVMatrix::addVector(NVMatrix& vec, float scaleVec) {
    addVector(vec, scaleVec, *this);
}

void NVMatrix::addVector(NVMatrix& vec, NVMatrix& target) {
    addVector(vec, 1, target);
}

void NVMatrix::subtractFromScalar(float scalar, NVMatrix& target) {
    target.resize(*this);
    kSubtractFromScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(_devData, scalar, target._devData,_numElements);
    cutilCheckMsg("Kernel execution failed");
}

void NVMatrix::subtractFromScalar(float scalar) {
    subtractFromScalar(scalar, *this);
}

void NVMatrix::addScalar(float scalar, NVMatrix& target) {
    target.resize(*this);
    kAddScalar<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(_devData, scalar, target._devData,_numElements);
    cutilCheckMsg("Kernel execution failed");
}

void NVMatrix::addScalar(float scalar) {
    addScalar(scalar, *this);
}

void NVMatrix::eltWiseMultByVector(NVMatrix& vec, NVMatrix& target) {
    assert(&target != &vec); // for now
    assert(vec.getNumRows() == 1 || vec.getNumCols() == 1);
    assert(vec.getNumRows() == _numRows || vec.getNumCols() == _numCols);
//    assert(&target != &vec);
    target.resize(*this);
    target._isTrans = _isTrans;

//    const unsigned int numThreads = numBlocks*numThreadsPerBlock;
    const unsigned int width = _isTrans ? _numRows : _numCols;
    const unsigned int height = _isTrans ? _numCols : _numRows;
    if(vec.getNumRows() == _numRows && !isTrans() || vec.getNumCols() == _numCols && isTrans()) {
        kMultByColVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(_devData, vec._devData, target._devData, width, height);
    } else {
        kMultByRowVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(_devData, vec._devData, target._devData, width, height);
    }
    cutilCheckMsg("Kernel execution failed");
}

void NVMatrix::eltWiseMultByVector(NVMatrix& vec) {
    eltWiseMultByVector(vec,  *this);
}

void NVMatrix::eltWiseDivideByVector(NVMatrix& vec) {
    eltWiseDivideByVector(vec,  *this);
}

void NVMatrix::eltWiseDivideByVector(NVMatrix& vec, NVMatrix& target) {
    NVMatrix* vecRecip = new NVMatrix(vec);
    vec.apply(NVMatrix::RECIPROCAL, *vecRecip);
    eltWiseMultByVector(*vecRecip, target);
    cudaThreadSynchronize();
    delete vecRecip;
}

void NVMatrix::eltWiseDivideByVector2(NVMatrix& vec) {
    eltWiseDivideByVector2(vec,  *this);
}

void NVMatrix::eltWiseDivideByVector2(NVMatrix& vec, NVMatrix& target) {
    assert(&target != &vec); // for now
    assert(vec.getNumRows() == 1 || vec.getNumCols() == 1);
    assert(vec.getNumRows() == _numRows || vec.getNumCols() == _numCols);
//    assert(&target != &vec);
    target.resize(*this);

//    const unsigned int numThreads = numBlocks*numThreadsPerBlock;
    const unsigned int width = _isTrans ? _numRows : _numCols;
    const unsigned int height = _isTrans ? _numCols : _numRows;
    if(vec.getNumRows() == _numRows && !isTrans() || vec.getNumCols() == _numCols && isTrans()) {
        kDivideByColVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(_devData, vec._devData, target._devData, width, height);
    } else {
        kDivideByRowVector<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(_devData, vec._devData, target._devData, width, height);
    }
    cutilCheckMsg("Kernel execution failed");
}

void NVMatrix::scale(float scale) {
    cublasSscal(_numElements, scale, _devData, 1);
    checkCublasError("cublasSscal failed.");
}

void NVMatrix::scale(float scale, NVMatrix& target) {
    target.resize(*this);
    target.copyFromDevice(*this);
    target.scale(scale);
}

/*
 * num threads per block is ignored when summing rows (axis=1) because
 * it has to be a power of 2.
 */
void NVMatrix::aggregate(int axis, NVMatrix& target, int numThreadsPerBlock, NVMatrix::AGGREGATIONS agg) {
    assert(&target != this);
    unsigned int width = _isTrans ? _numRows : _numCols;
    const int height = _isTrans ? _numCols : _numRows;

    target.setTrans(_isTrans);
    assert(width > 0);
    assert(height > 0);
    if(axis == 0 && !_isTrans || axis == 1 && _isTrans) { //col sum
        target.resize(!_isTrans ? 1 : _numRows, !_isTrans ? _numCols : 1);
        const unsigned int numBlocks = (width + numThreadsPerBlock - 1) / numThreadsPerBlock;
        assert(numBlocks * numThreadsPerBlock >= width);
        assert(numBlocks < NUM_BLOCKS_MAX);
//        target.resize(1, width);
        if(agg == NVMatrix::MAX) {
            kDumbMaxCols<<<numBlocks,numThreadsPerBlock>>>(_devData, target._devData, width, height);
        } else if(agg == NVMatrix::SUM) {
            kDumbSumCols<<<numBlocks,numThreadsPerBlock>>>(_devData, target._devData, width, height);
        }
        cutilCheckMsg("Kernel execution failed");
    } else { // row sum
        target.resize(_isTrans ? 1 : _numRows, _isTrans ? _numCols : 1);
        if (width > 1) {
            NVMatrix *prevSum = this;

            while (prevSum->getLeadingDim()  > 1) {
                int numBlocksX, numBlocksY, numThreadsX, numThreadsY;
                bool doLinearAgg = height >= 16384;
//                doQuickAgg = !doQuickAgg;

                if(doLinearAgg) { // call the special short aggregation functions
                    numBlocksX = 1;
                    numBlocksY = DIVUP(height, AGG_SHORT_ROWS_THREADS_Y*AGG_SHORT_ROWS_LOOPS_Y);
                    numThreadsX = AGG_SHORT_ROWS_THREADS_X;
                    numThreadsY = AGG_SHORT_ROWS_THREADS_Y;
                    while(numBlocksY > NUM_BLOCKS_MAX) {
                        numBlocksY = DIVUP(numBlocksY,2);
                        numBlocksX *= 2;
                    }
                } else {
                    numThreadsX = width <= 64 ? 32 : (width <= 128 ? 64 : (width <= 256 ? 128 : (width <= 512 ? 256 : 512)));
                    numThreadsY = 1;
                    numBlocksX = DIVUP(width, 2*numThreadsX);
                    numBlocksY = std::min(height, NUM_BLOCKS_MAX);
                }

                dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
                assert(numBlocksX <= NUM_BLOCKS_MAX);
                assert(numBlocksY <= NUM_BLOCKS_MAX);
//                printf("%d %d %d %d %d \n", numThreadsX, numThreadsY, numBlocksX, numBlocksY, numBlocksZ);

                NVMatrix *nvSumAccum = target.getFollowingDim() == height && target.getLeadingDim() == numBlocksX ? &target : new NVMatrix(height, numBlocksX, false);
//                printf("target size: %dx%d\n", target.getNumRows(), target.getNumCols());
//                printf("liear agg: %d, width: %d, height: %d\n", doLinearAgg, width, height);
//                printf("accum is target: %d\n", nvSumAccum == &target);
                if(agg == NVMatrix::MAX) {
                    if(doLinearAgg) {
                        if(width <= 16) {
                            if(width <= 4) {
                                kAggShortRows<AGG_MAX, 1, 4><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else if(width <= 8) {
                                kAggShortRows<AGG_MAX, 1, 8><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else if(width <= 12) {
                                kAggShortRows<AGG_MAX, 1, 12><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else {
                                kAggShortRows<AGG_MAX, 1, 16><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            }

                        } else if(width <= 32) {
                            kAggShortRows<AGG_MAX, 2, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else if(width <= 48){
                            kAggShortRows<AGG_MAX, 3, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else if(width <= 64){
                            kAggShortRows<AGG_MAX, 4, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else {
                            kAggShortRows2<AGG_MAX><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        }
                    } else if(width <= 64) {
                        kMaxRows<32><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    } else if(width <= 128) {
                        kMaxRows<64><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    } else if(width <= 256) {
                        kMaxRows<128><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    } else if(width <= 512) {
                        kMaxRows<256><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    } else {
                        kMaxRows<512><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                   width, height, nvSumAccum->getLeadingDim());
                    }
                } else if(agg == NVMatrix::SUM) {
                    if(doLinearAgg) {
                        if(width <= 16) {
                            if(width <= 4) {
                                kAggShortRows<AGG_SUM, 1, 4><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else if(width <= 8) {
                                kAggShortRows<AGG_SUM, 1, 8><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else if(width <= 12) {
                                kAggShortRows<AGG_SUM, 1, 12><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            } else {
                                kAggShortRows<AGG_SUM, 1, 16><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                            }
                        } else if(width <= 32) {
                            kAggShortRows<AGG_SUM, 2, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else if(width <= 48) {
                            kAggShortRows<AGG_SUM, 3, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else if(width <= 64){
                            kAggShortRows<AGG_SUM, 4, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        } else {
                            kAggShortRows2<AGG_SUM><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,width, height);
                        }
                    } else if (width <= 64) {
                        kSumRows<32><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                width, height, nvSumAccum->getLeadingDim());
                    } else if (width <= 128) {
                        kSumRows<64><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                width, height, nvSumAccum->getLeadingDim());
                    } else if (width <= 256) {
                        kSumRows<128><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                width, height, nvSumAccum->getLeadingDim());
                    } else if (width <= 512) {
                        kSumRows<256><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                width, height, nvSumAccum->getLeadingDim());
                    } else {
                        kSumRows<512><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                width, height, nvSumAccum->getLeadingDim());
                    }
                }
                cutilCheckMsg("Kernel execution failed");
                cudaThreadSynchronize();
                width = numBlocksX;

                if (prevSum != this) {
                    delete prevSum;
                }
                prevSum = nvSumAccum;
            }
//            if (_isTrans) {
//                prevSum->_numCols = prevSum->_numRows;
//                prevSum->_numRows = 1;
//            }
//            target.copyFromDevice(*prevSum);
//            delete prevSum;
        } else {
            target.resize(*this);
            target.copyFromDevice(*this);
        }
    }
}

void NVMatrix::max(int axis, NVMatrix& target) {
    if(axis == 0 && !_isTrans || axis == 1 && _isTrans) {
        aggregate(axis, target, NUM_SUM_COLS_THREADS_PER_BLOCK, NVMatrix::MAX);
    } else {
        aggregate(axis, target, NUM_SUM_ROWS_THREADS_PER_BLOCK, NVMatrix::MAX);
    }
}

NVMatrix& NVMatrix::max(int axis) {
    NVMatrix *sumVec = new NVMatrix();
    max(axis, *sumVec);
    return *sumVec;
}

void NVMatrix::sum(int axis, NVMatrix& target) {
    assert(axis == 0 || axis == 1);
    if(axis == 0 && !_isTrans || axis == 1 && _isTrans) {
        aggregate(axis, target, NUM_SUM_COLS_THREADS_PER_BLOCK, NVMatrix::SUM);
    } else {
        aggregate(axis, target, NUM_SUM_ROWS_THREADS_PER_BLOCK, NVMatrix::SUM);
    }
}

NVMatrix& NVMatrix::sum(int axis) {
    NVMatrix *sumVec = new NVMatrix();
    sum(axis, *sumVec);
    return *sumVec;
}

float NVMatrix::sum() {
    WARN("Summing over all matrix elements first performs a sum over all columns. If your matrix has few columns, this is inefficient.");

    NVMatrix devSum = NVMatrix();
    sum(_isTrans && _numRows > _numCols || !_isTrans && _numRows < _numCols ? 1 : 0, devSum);
    Matrix hostSum = Matrix(devSum._numRows, devSum._numCols);
    cudaThreadSynchronize();
    devSum.copyToHost(hostSum);
    return hostSum.sum();
}

void NVMatrix::print(int startRow, int rows, int startCol, int cols) const {
    cudaThreadSynchronize();
    Matrix* hm = new Matrix(_numRows, _numCols);
    copyToHost(*hm);
    hm->print(startRow, rows, startCol, cols);
    delete hm;
}

void NVMatrix::print(int rows, int cols) const {
    print(0, rows, 0, cols);
}

//========================================================
// NVMatrix but initialized with zeros instead of whatever
// happens to be in memory.
//========================================================

NVZeroMatrix::NVZeroMatrix(int numRows, int numCols, bool isTrans) : NVMatrix(numRows, numCols, isTrans) {
    apply(NVMatrix::ZERO);
}

NVZeroMatrix::NVZeroMatrix(Matrix& like) : NVMatrix(like.getNumRows(), like.getNumCols()) {
    apply(NVMatrix::ZERO);
}

NVZeroMatrix::NVZeroMatrix(NVMatrix& like) : NVMatrix(like.getNumRows(), like.getNumCols()) {
    apply(NVMatrix::ZERO);
}
