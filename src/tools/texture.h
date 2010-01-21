/* Copyright 2008 NVIDIA Corporation.  All Rights Reserved */
#ifndef __TEXTURE_H__
#define __TEXTURE_H__

#include <cstdio>
#include "cuda.h"
#include "cuv_general.hpp"


/*
 * These textures are (optionally) used to cache the 'x' vector in y += A*x
 */
texture<float> tex_x_float;

// Use int2 to pull doubles through texture cache
size_t bind_x(const float * x, const unsigned int len)
{   
	size_t offset;
	cuvSafeCall(cudaBindTexture(&offset, tex_x_float, (const void *)x, sizeof(float)*len));
	cuvAssert(offset % sizeof(float) == 0 );
	return offset/sizeof(float);
}

void unbind_x(const float * x)
{   cuvSafeCall(cudaUnbindTexture(tex_x_float)); }
// Note: x is unused, but distinguishes the two functions

template <bool UseCache>
__inline__ __device__ float fetch_x(const float* x, const int& i)
{
    if (UseCache)
        return tex1Dfetch(tex_x_float, i);
    else
        return x[i];
}
#endif /* __TEXTURE_H__ */
