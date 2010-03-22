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
texture<uchar4> tex_x_uchar4;
texture<uchar1> tex_x_uchar1;

inline size_t bind_x(const uchar1 * x, const unsigned int len)
{   
	size_t offset;
	cuvSafeCall(cudaBindTexture(&offset, tex_x_uchar1, (const void *)x, sizeof(uchar1)*len));
	cuvAssert(offset % sizeof(uchar1) == 0 );
	return offset/sizeof(uchar1);
}
inline size_t bind_x(const uchar4 * x, const unsigned int len)
{   
	size_t offset;
	cuvSafeCall(cudaBindTexture(&offset, tex_x_uchar4, (const void *)x, sizeof(uchar4)*len));
	cuvAssert(offset % sizeof(uchar4) == 0 );
	return offset/sizeof(uchar4);
}
inline size_t bind_x(const float * x, const unsigned int len)
{   
	size_t offset;
	cuvSafeCall(cudaBindTexture(&offset, tex_x_float, (const void *)x, sizeof(float)*len));
	cuvAssert(offset % sizeof(float) == 0 );
	return offset/sizeof(float);
}

// Note: x is unused, but distinguishes the functions
inline void unbind_x(const float * x)
{   cuvSafeCall(cudaUnbindTexture(tex_x_float)); }
inline void unbind_x(const uchar4 * x)
{   cuvSafeCall(cudaUnbindTexture(tex_x_uchar4)); }
inline void unbind_x(const uchar1 * x)
{   cuvSafeCall(cudaUnbindTexture(tex_x_uchar1)); }

template <bool UseCache>
inline __device__ float fetch_x(const float* x, const int& i)
{
    if (UseCache) return tex1Dfetch(tex_x_float, i);
    else          return x[i];
}
template <bool UseCache>
inline __device__ uchar4 fetch_x(const uchar4* x, const int& i)
{
    if (UseCache) return tex1Dfetch(tex_x_uchar4, i);
    else          return x[i];
}
template <bool UseCache>
inline __device__ uchar1 fetch_x(const uchar1* x, const int& i)
{
    if (UseCache) return tex1Dfetch(tex_x_uchar1, i);
    else          return x[i];
}



#endif /* __TEXTURE_H__ */
