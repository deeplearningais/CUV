/* Copyright 2008 NVIDIA Corporation.  All Rights Reserved */

#pragma once

#include "cuda.h"
#include "cuv_general.hpp"


/*
 * These textures are (optionally) used to cache the 'x' vector in y += A*x
 */
texture<float,1> tex_x_float;
texture<int2,1>  tex_x_double;

// Use int2 to pull doubles through texture cache
void bind_x(const float * x, const unsigned int len)
{   
	cudaBindTexture(NULL, tex_x_float, (const void *)x, sizeof(float)*len);
	//cuvSafeCall(cudaBindTexture(NULL, tex_x_float, x));   
}

void bind_x(const double * x)
{   cuvSafeCall(cudaBindTexture(NULL, tex_x_double, x));   }

void unbind_x(const float * x)
{   cuvSafeCall(cudaUnbindTexture(tex_x_float)); }
void unbind_x(const double * x)
{   cuvSafeCall(cudaUnbindTexture(tex_x_double)); }
// Note: x is unused, but distinguishes the two functions

template <bool UseCache>
__inline__ __device__ float fetch_x(const int& i, const float * x)
{
    if (UseCache)
        return tex1Dfetch(tex_x_float, i);
    else
        return x[i];
}

#if !defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)
template <bool UseCache>
__inline__ __device__ double fetch_x(const int& i, const double * x)
{
    if (UseCache){
        int2 v = tex1Dfetch(tex_x_double, i);
        return __hiloint2double(v.y, v.x);
    } else {
        return x[i];
    }
}
#endif // !defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)

