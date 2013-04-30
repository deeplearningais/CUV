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





#include <string>
#include <cmath>
#include <iostream>

#include "random.hpp"

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)

namespace cuv{
    __global__ void setup_kernel(curandState* state, unsigned long long  seed){
    const uint tidx = NUM_RND_THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    /* Each thread gets same seed, a different sequence number,
     no offset */
    curand_init(seed, tidx, 0, &state[tidx]);
    };
	// Initialize seeds for the Mersenne Twister
	static bool* g_mersenne_twister_initialized;
    static curandState** g_rnd_dev_state = NULL;
	void initialize_mersenne_twister_seeds(unsigned int seed) {
        if(g_rnd_dev_state==NULL){
            int cnt;
            cuvSafeCall(cudaGetDeviceCount(&cnt));
            g_rnd_dev_state = new curandState* [cnt];
            g_mersenne_twister_initialized = new bool[cnt];
            for(int i=0;i<cnt;i++){
                g_rnd_dev_state[i] = NULL;
                g_mersenne_twister_initialized[i]=false;
            }
        }
        int dev;
        cuvSafeCall(cudaGetDevice(&dev));
        if(g_rnd_dev_state[dev]==NULL){
            cuvSafeCall(cudaMalloc((void **)&g_rnd_dev_state[dev], NUM_RND_STREAMS * sizeof(curandState)));
        }
        setup_kernel<<<NUM_RND_BLOCKS, NUM_RND_THREADS_PER_BLOCK>>>(g_rnd_dev_state[dev], 1 + seed*2); // so there's no chance it'll be correlated with the other one
        cuvSafeCall(cudaThreadSynchronize());
		g_mersenne_twister_initialized[dev] = true;
	}
	void deinit_rng(unsigned int seed) {
        int dev;
        cuvSafeCall(cudaGetDevice(&dev));
        cuvSafeCall(cudaFree(g_rnd_dev_state[dev]));
        g_rnd_dev_state[dev] = NULL;
		g_mersenne_twister_initialized[dev] = false;
	}
    
    struct uf_binarize{
        public:
            __device__ inline float operator()(float f, curandState* state){
                return f>curand_uniform(state);
            }
    };
    struct uf_uniform{
        public:
            __device__ inline float operator()(float f, curandState* state){
                return curand_uniform(state);
            }
    };
    struct uf_gaussian{
        private: 
            float m_mean, m_std;
        public:
            uf_gaussian(float mean, float std):m_mean(mean),m_std(std){}
            __device__ inline float operator()(float f, curandState* state){
                return m_mean+m_std*curand_normal(state);
            }
    };
    struct uf_add_gaussian{
        private: 
            float m_std;
        public:
            uf_add_gaussian(float std):m_std(std){}
            __device__ inline float operator()(float f, curandState* state){
                return f + m_std*curand_normal(state);
            }
    };

    template<class Op>
    __global__ void unary_rng_kernel(float* dst, const float* src, curandState* state, unsigned int size, Op op){
        const unsigned int tidx = NUM_RND_THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
        curandState localState = state[tidx];
        for(unsigned int i=tidx;i<size;i+=NUM_RND_STREAMS){
            dst[i] = op(src[i], &localState);
        }
        state[tidx] = localState;
    };

    template<class Op>
        void
    call_unary_rng_kernel(tensor<float,dev_memory_space>& dst, tensor<float,dev_memory_space>& src, const Op& op){
        int dev;
        cuvSafeCall(cudaGetDevice(&dev));
		cuvAssert(g_mersenne_twister_initialized[dev]);

        unary_rng_kernel<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(dst.ptr(), src.ptr(),g_rnd_dev_state[dev], dst.size(), op);
		cuvSafeCall(cudaThreadSynchronize());
    }

	template<>
	void rnd_binarize(tensor<float,dev_memory_space>& v){
		cuvAssert(v.ptr());
		call_unary_rng_kernel(v,v,uf_binarize());
	}
	template<>
	void rnd_binarize(tensor<float,host_memory_space>& v){
	   cuvAssert(v.ptr());
	   tensor<float,host_memory_space>::value_type* ptr = v.ptr();
	   for(int i=0;i<v.size();i++)
		   *ptr++ = ((float)rand()/RAND_MAX) < *ptr;
	}
        template<>
	void rnd_binarize(tensor<float,host_memory_space,column_major>& v){
            rnd_binarize(*reinterpret_cast<tensor<float,host_memory_space>* >(&v));
        }
        template<>
	void rnd_binarize(tensor<float,dev_memory_space,column_major>& v){
            rnd_binarize(*reinterpret_cast<tensor<float,dev_memory_space>* >(&v));
        }
	template<>
	void fill_rnd_uniform(tensor<float,host_memory_space>& v){
	   cuvAssert(v.ptr());
	   tensor<float,host_memory_space>::value_type* ptr = v.ptr();
       unsigned int size = v.size();
	   for(unsigned int i=0;i<size;i++)
		   *ptr++ = ((float)rand()/RAND_MAX);
	}
	template<>
	void fill_rnd_uniform(tensor<float,dev_memory_space>& v){
		cuvAssert(v.ptr());
		call_unary_rng_kernel(v,v,uf_uniform());
	}
        template<>
	void fill_rnd_uniform(tensor<float,host_memory_space,column_major>& v){
            fill_rnd_uniform(*reinterpret_cast<tensor<float,host_memory_space>* >(&v));
        }
        template<>
	void fill_rnd_uniform(tensor<float,dev_memory_space,column_major>& v){
            fill_rnd_uniform(*reinterpret_cast<tensor<float,dev_memory_space>* >(&v));
        }
        double norminv(double q) {
            if(q == .5)
                return 0;

            q = 1.0 - q;

            double p = (q > 0.0 && q < 0.5) ? q : (1.0 - q);
            double t = sqrt(log(1.0 / pow(p, 2.0)));

            double c0 = 2.515517;
            double c1 = 0.802853;
            double c2 = 0.010328;

            double d1 = 1.432788;
            double d2 = 0.189269;
            double d3 = 0.001308;

            double x = t - (c0 + c1 * t + c2 * pow(t, 2.0)) /
                (1.0 + d1 * t + d2 * pow(t, 2.0) + d3 * pow(t, 3.0));

            if(q > .5)
                x *= -1.0;

            return x;
        }
	template<>
	void add_rnd_normal(tensor<float,host_memory_space>& v, const float& std){
	   cuvAssert(v.ptr());
	   tensor<float,host_memory_space>::value_type* ptr = v.ptr();
       unsigned int size = v.size();
	   for(unsigned int i=0;i<size;i++)
		   *ptr++ += std*norminv(drand48());
	}
	template<>
	void add_rnd_normal(tensor<float,dev_memory_space>& v, const float& std){
        call_unary_rng_kernel(v,v,uf_add_gaussian(std)); 
	}
        template<>
	void add_rnd_normal(tensor<float,dev_memory_space,column_major>& v, const float& std){
            add_rnd_normal(*reinterpret_cast<tensor<float,dev_memory_space>* >(&v),std);
        }
        template<>
	void add_rnd_normal(tensor<float,host_memory_space,column_major>& v, const float& std){
            add_rnd_normal(*reinterpret_cast<tensor<float,host_memory_space>* >(&v),std);
        }

} // cuv
