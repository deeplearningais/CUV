/*
 * nvmatrix.h
 *
 *  Created on: 20-Jan-2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#ifndef NVMATRIX_H_
#define NVMATRIX_H_

//#define RND_MULTIPLIERS_FILE ("rnd_multipliers_32bit.txt")

#ifndef RND_MULTIPLIERS_FILE
#define RND_MULTIPLIERS_FILE ("rnd_multipliers_32bit.txt")
#endif

#include <cublas.h>
#include <cutil_inline.h>

#include "matrix.h"
#include "nvmatrix_kernel.cuh"

#ifdef WARNINGS
#define WARN(msg) printf("WARN: File %s, line %d: %s\n", __FILE__, __LINE__, msg);
#else
#define WARN(msg) ;
#endif

class NVMatrix {
private:
    unsigned int _numCols, _numRows;
    unsigned int _numElements;
    float* _devData;
    bool _isTrans;
    bool _ownsData;
    static cudaDeviceProp deviceProps;

    static unsigned int hostRndMults[NUM_RND_STREAMS];
    static bool rndInitialized;
    static unsigned int *devRndMults;
    static unsigned long long *devRndWords;

    static inline void checkCublasError(const char* msg) {
        cublasStatus status = cublasGetError();
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, msg, NULL);
            exit(EXIT_FAILURE);
        }
    }

    inline unsigned int getDefaultNumThreadsPerBlock() {
        return deviceProps.maxThreadsPerBlock;
    }

    /*
     * WARNING: this is probably a crappy default! test it out before using.
     */
    inline unsigned int getDefaultNumBlocks() {
        return std::min(int(ceil(_numElements / double(getDefaultNumThreadsPerBlock()))), NUM_BLOCKS_MAX);
    }

    inline char getTransChar() const {
        /*
         * not a typo! return opposite character because a
         * non-transposed krizhevsky matrix is in row-major order while a non-transposed
         * cublas matrix is in column-major order.
         */
        return _isTrans ? 'n' : 't';
    }


    inline unsigned int getNumRowsBackEnd() const {
        return _isTrans ? _numCols : _numRows;
    }

    void _init(unsigned int numRows, unsigned int numCols);

public:
    enum FUNCTIONS {LOG, LOGISTIC1, LOGISTIC2, EXP, SQUARE, SQRT, ZERO, RECIPROCAL};
    enum AGGREGATIONS {SUM, MAX, MIN};
    NVMatrix();
    NVMatrix(bool isTrans);
    NVMatrix(int numRows, int numCols, bool isTrans=true);
    NVMatrix(const Matrix& like, bool copy);
    NVMatrix(const NVMatrix& like, bool copy);
    NVMatrix(const NVMatrix& like);
    NVMatrix(const Matrix& like);
    NVMatrix(float* devData, int numRows, int numCols, bool isTrans);
    ~NVMatrix();

    static void initDeviceProps();
    static void initRandom(unsigned int seed);
    static void destroyRandom();

    /*
     * DO NOT DEREFERENCE IN HOST CODE! This is a device memory pointer.
     */
    inline float* getCellPtr(int i, int j) const {
        if (_isTrans) {
            return &_devData[j * _numRows + i];
        }
        return &_devData[i * _numCols + j];
    }

    inline bool isSameDims(const Matrix& m) const {
        return m.getNumRows() == _numRows && m.getNumCols() == _numCols;
    }

    inline bool isSameDims(const NVMatrix& m) const {
        return m.getNumRows() == _numRows && m.getNumCols() == _numCols;
    }

    inline int getNumRows() const {
        return _numRows;
    }

    inline int getNumCols() const {
        return _numCols;
    }

    inline unsigned int getLeadingDim() const {
        return _isTrans ? _numRows : _numCols;
    }

    inline unsigned int getFollowingDim() const {
        return !_isTrans ? _numRows : _numCols;
    }

    /*
     * FALSE:    Row-major order.
     * TRUE:     Column-major order.
     */
    inline bool isTrans() const {
        return _isTrans;
    }

    inline bool isView() const {
        return !_ownsData;
    }

    inline float* getDevData() const {
        return _devData;
    }

    inline unsigned int getNumElements() const {
        return _numElements;
    }

    /*
     * Only use if you know what you're doing!
     * Does not actually transpose matrix.
     */
    inline void setTrans(bool trans) {
        _isTrans = trans;
    }

    void copyFromHost(const Matrix& hostMatrix);
    void copyFromHost(const Matrix& hostMatrix, bool resizeDeviceMatrix);
    void copyToHost(Matrix& hostMatrix) const;
    void copyFromDevice(const NVMatrix & devMatrix);
    void copyFromDevice(const NVMatrix& devMatrix, bool resizeTarget);
    void addProduct(const NVMatrix& a, const NVMatrix &b, float scaleThis, float scaleAB);
    void addProduct(const NVMatrix& a, const NVMatrix &b);
    void rightMult(const NVMatrix &b, float scaleAB, NVMatrix &target) const;
    void rightMult(const NVMatrix &b, NVMatrix &target) const;
    void rightMult(const NVMatrix &b, float scaleAB);
    void randomizeUniform();
    void addGaussianNoise(float stdev);
    void addGaussianNoise();
    void randomizeGaussian();
    void randomizeGaussian(float stdev);
    void binarizeProbs();
    void biggerThanScalar(float scalar, NVMatrix& target);
    void biggerThanScalar(float scalar);

    void biggerThan(NVMatrix& m, NVMatrix& target, int numBlocks=NUM_APPLY_BLOCKS, int numThreadsPerBlock=NUM_APPLY_THREADS_PER_BLOCK);
    void biggerThan(NVMatrix& m, int numBlocks=NUM_APPLY_BLOCKS, int numThreadsPerBlock=NUM_APPLY_THREADS_PER_BLOCK);

    void _checkBounds(int startRow, int endRow, int startCol, int endCol) const;
    NVMatrix& slice(int startRow, int endRow, int startCol, int endCol) const;
    void slice(int startRow, int endRow, int startCol, int endCol, NVMatrix& target) const;
    NVMatrix& sliceRows(int startRow, int endRow) const;
    void sliceRows(int startRow, int endRow, NVMatrix& target) const;
    NVMatrix& sliceCols(int startCol, int endCol) const;
    void sliceCols(int startCol, int endCol, NVMatrix& target) const;

    void apply(NVMatrix::FUNCTIONS f, NVMatrix& target, int numBlocks=NUM_APPLY_BLOCKS, int numThreadsPerBlock=NUM_APPLY_THREADS_PER_BLOCK);
    void apply(NVMatrix::FUNCTIONS f, int numBlocks=NUM_APPLY_BLOCKS, int numThreadsPerBlock=NUM_APPLY_THREADS_PER_BLOCK);

    bool resize(int numRows, int numCols);
    bool resize(const NVMatrix &like);
    bool resize(const Matrix &like);
    void reshape(int numRows, int numCols);
    NVMatrix& reshaped(int numRows, int numCols);

    void copy(NVMatrix &dest, int srcStartRow, int srcEndRow,
            int srcStartCol, int srcEndCol, int destStartRow, int destStartCol,
            int numBlocks, int numThreadsPerBlock) const;
    void copy(NVMatrix &dest, int srcStartRow, int srcEndRow, int srcStartCol, int srcEndCol,
                        int destStartRow, int destStartCol) const;

    void add(NVMatrix& b, float scaleA, float scaleB, NVMatrix& target);
    void add(NVMatrix& b, float scaleB, NVMatrix& target);
    void add(NVMatrix& b, NVMatrix& target);
    void add(NVMatrix& b, float scaleB);
    void add(NVMatrix& b, float scaleA, float scaleB);
    void add(NVMatrix& b);
    void addScalar(float scalar, NVMatrix& target);
    void addScalar(float scalar);
    void eltWiseMult(NVMatrix& b);
    void eltWiseMult(NVMatrix& b, NVMatrix& target);
    void eltWiseDivide(NVMatrix& b);
    void eltWiseDivide(NVMatrix& b, NVMatrix& target);
    void subtractFromScalar(float scalar, NVMatrix& target);
    void subtractFromScalar(float scalar);
    void squaredDiff(NVMatrix& b);
    void squaredDiff(NVMatrix& b, NVMatrix& target);
    void addSum(NVMatrix& b, NVMatrix& c, float scaleThis, float scaleB, float scaleC);
    void subtract(NVMatrix& b, NVMatrix& target);
    void subtract(NVMatrix& b);
    void addVector(NVMatrix& vec, float scaleVec, NVMatrix& target, int numBlocks=NUM_APPLY_BLOCKS, int numThreadsPerBlock=NUM_APPLY_THREADS_PER_BLOCK);
    void addVector(NVMatrix& vec);
    void addVector(NVMatrix& vec, float scaleVec);
    void addVector(NVMatrix& vec, NVMatrix& target);
    void eltWiseMultByVector(NVMatrix& vec, NVMatrix& target);
    void eltWiseMultByVector(NVMatrix& vec);
    void eltWiseDivideByVector(NVMatrix& vec, NVMatrix& target);
    void eltWiseDivideByVector(NVMatrix& vec);
    void eltWiseDivideByVector2(NVMatrix& vec, NVMatrix& target);
    void eltWiseDivideByVector2(NVMatrix& vec);
    void tile(int timesY, int timesX, NVMatrix& target, int numBlocks=NUM_APPLY_BLOCKS, int numThreadsPerBlock=NUM_APPLY_THREADS_PER_BLOCK);
    void scale(float scale);
    void scale(float scale, NVMatrix& target);
    void aggregate(int axis, NVMatrix& target, int numThreadsPerBlock, NVMatrix::AGGREGATIONS agg);
    void sum(int axis, NVMatrix& target);
    NVMatrix& sum(int axis);
    float sum();
    void max(int axis, NVMatrix& target);
    NVMatrix& max(int axis);

    NVMatrix& getTranspose();
    void flipTrans();

    void print(int startRow, int rows, int startCol, int cols) const;
    void print(int rows, int cols) const;

};

class NVZeroMatrix : public NVMatrix {
public:
    NVZeroMatrix(int numRows, int numCols, bool isTrans=true);
    NVZeroMatrix(Matrix& like);
    NVZeroMatrix(NVMatrix& like);

    virtual ~NVZeroMatrix() {

    }
};

#endif /* NVMATRIX_H_ */
