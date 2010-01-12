/*
 * convCPU.h
 *
 *  Created on: Oct 31, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#ifndef CONVCPU_H_
#define CONVCPU_H_

inline float dotCPU(float* img, float* filter, int imgSize, int filterSize, int y, int x);
void convCPU(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgs, int numFilters);
void convColorCPU(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgs, int numFilters);
void conv2CPU(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgs, int numFilters);
void conv2ColorCPU(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgs, int numFilters);
void rotate180CPU(float* filters, float* targets, int filterSize, int numFilters);
void padZerosCPU(float* images, float* targets, int imgSize, int numImages, int paddingSize);
void conv3CPU(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgs, int numFilters);
void conv3ColorCPU(float* imgs, float* filters, float* targets, int imgSize, int filterSize, int numImgs, int numFilters);
void subsampleCPU(float* images, float* targets, int imgSize, int factor, int numImgs);
void supersampleCPU(float* images, float* targets, int imgSize, int factor, int numImgs, bool trans);
void gridToMatrixCPU(float* images, float* targets, int imgSize, int factor, int numImgs);
void matrixToGridCPU(float* images, float* targets, int imgSize, int factor, int numImgs);
void sampleMultinomialCPU(float* multi, float* randoms, float* targets, int multinomials, int nomials);
#endif /* CONVCPU_H_ */
