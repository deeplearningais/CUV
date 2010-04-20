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
