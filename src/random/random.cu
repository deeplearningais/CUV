#include <string>
#include <cmath>
#include <iostream>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random.hpp>

#include <cuda.h>
#include <cutil_inline.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/generate.h>

 
#include <thrust/transform_reduce.h>

// Mersenne Twister code reproduced & modified from: 
//										http://svn.jcornwall.me.uk/applications/MersenneTwisterGPU/
//										http://www.jcornwall.me.uk/2009/04/mersenne-twisters-in-cuda/
#include <cassert>
#include <cstdio>
#include <vector>

#include <cuv_general.hpp>
#include <dev_vector.hpp>
#include <host_vector.hpp>
#include "random.hpp"


// Old RNG 

/*__global__ void kSeedRandom(unsigned int* rndMults, unsigned long long* rndWords, unsigned int seed) {*/
/*    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;*/
        
/*    // The initial x is the seed and the initial carry is 1*/
/*    unsigned long long rndWord = ((unsigned long long)seed << 32) + 1;*/
/*    const unsigned int rndMult = rndMults[idx]; */
    /*
     * Run the chain for a few steps so that all the streams have a chance
     * to differentiate. They start out generating similar random numbers
     * because all the multipliers are similar. 
     */
/*    for(unsigned int i = 0; i < NUM_RND_BURNIN; i++) {*/
/*        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);*/
/*    }*/
/*    rndWords[idx] = rndWord;*/
/*}  */

/*__global__ void kRandomUniform(unsigned int* rndMults, unsigned long long* rndWords, float* gData, unsigned int numElements) {*/
/*    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;*/
/*    unsigned long long rndWord = rndWords[idx]; */
/*    const unsigned int rndMult = rndMults[idx];*/
   
/*    for(unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {*/
/*        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);*/
/*        gData[i] = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;*/
/*    }*/
/*    rndWords[idx] = rndWord;*/
/*}       */

/*
 * TODO: modify to take mean/stdev 
 */     
/*__global__ void kRandomGaussian(unsigned int* rndMults, unsigned long long* rndWords, float* gData, unsigned int numElements) {*/
/*    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;*/
/*    unsigned long long rndWord = rndWords[idx];*/
/*    const unsigned int rndMult = rndMults[idx];*/

/*    float rnd1, rnd2, R, T;*/
/*    for(unsigned int i = idx; i < numElements; i += 2*NUM_RND_STREAMS) {*/
/*        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);*/
/*        rnd1 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;*/
/*        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);*/
/*        rnd2 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;*/
/*        T = 2 * M_PI * rnd2;*/
/*        R = sqrtf(-2 * __logf(rnd1));*/
/*        gData[i] = R * __cosf(T);*/
/*        if (i + NUM_RND_STREAMS < numElements)*/
/*            gData[i + NUM_RND_STREAMS] = R * __sinf(T);*/
/*    }*/
/*    rndWords[idx] = rndWord;*/
/*}*/

/*  
 * TODO: modify to take mean
 */     
/*__global__ void kAddGaussianNoise(unsigned int* rndMults, unsigned long long* rndWords, float* gData, float stdev, unsigned int numElements) {*/
/*    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;*/
/*    unsigned long long rndWord = rndWords[idx];*/
/*    const unsigned int rndMult = rndMults[idx];*/
        
/*    float rnd1, rnd2, R, T;*/
/*    for(unsigned int i = idx; i < numElements; i += 2*NUM_RND_STREAMS) {*/
/*        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);*/
/*        rnd1 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;*/
/*        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);*/
/*        rnd2 = (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;*/
/*        T = 2 * M_PI * rnd2;*/
/*        R = sqrtf(-2 * __logf(rnd1));*/
/*        gData[i] += stdev * R * __cosf(T);*/
/*        if (i + NUM_RND_STREAMS < numElements)*/
/*            gData[i + NUM_RND_STREAMS] += stdev * R * __sinf(T);*/
/*    }*/
/*    rndWords[idx] = rndWord;*/
/*}*/



/*template<class T>*/
/*void MatrixTools<T>::init_RNG(unsigned int seed, const char* fn) {*/
/*    assert(!sRndInitialized);*/
/*    std::ifstream inFile;*/
/*    inFile.open(fn);*/
/*    if(!inFile) {*/
/*        std::cerr << "Unable to open file " << RND_MULTIPLIERS_FILE << std::endl;*/
/*        exit(EXIT_FAILURE);*/
/*    }*/
    
/*    int numRead = 0;*/
/*    unsigned int mult;*/
/*    shRndMults = new unsigned int[NUM_RND_STREAMS];*/
/*    while(numRead < NUM_RND_STREAMS) {*/
/*        if(!(inFile >> mult)) {*/
/*            std::cerr << "Not enough numbers in file " << RND_MULTIPLIERS_FILE << std::endl;*/
/*            exit(EXIT_FAILURE);*/
/*        }*/
/*        shRndMults[numRead] = mult;*/
/*        numRead++;*/
/*    }*/
/*    inFile.close();*/

/*    cutilSafeCall(cudaMalloc((void **)&sdRndMults,   NUM_RND_STREAMS * sizeof(unsigned int)));*/
/*    cutilSafeCall(cudaMalloc((void **)&sdRndWords,   NUM_RND_STREAMS * sizeof(unsigned long long)));*/
/*    cutilSafeCall(cudaMemcpy(sdRndMults, shRndMults, NUM_RND_STREAMS * sizeof(unsigned int), cudaMemcpyHostToDevice));*/

/*    kSeedRandom<<<NUM_RND_BLOCKS, NUM_RND_THREADS_PER_BLOCK>>>(sdRndMults, sdRndWords, seed);*/
/*    cutilSafeCall(cudaThreadSynchronize());*/
/*    checkCudaError("Kernel execution failed");*/
/*    sRndInitialized = true;*/
/*}*/

/*template<class T>*/
/*void MatrixTools<T>::init_rnd_uniform(T& m) {*/
/*    assert(sRndInitialized);*/
/*    assert(m.getDev());*/
/*    kRandomUniform<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(sdRndMults, sdRndWords, m.getDev(),m.n());*/
/*    cutilSafeCall(cudaThreadSynchronize());*/
/*    checkCudaError("Kernel execution failed");*/
/*}*/
/*template<class T>*/
/*void MatrixTools<T>::init_rnd_gaussian(T& m) {*/
/*    assert(sRndInitialized);*/
/*    assert(m.getDev());*/
/*    kRandomGaussian<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(sdRndMults, sdRndWords, m.getDev(),m.n());*/
/*    cutilSafeCall(cudaThreadSynchronize());*/
/*    checkCudaError("Kernel execution failed");*/
/*}*/

/*template<class T>*/
/*void MatrixTools<T>::add_gaussian_noise(T& m, float stddev) {*/
/*    assert(sRndInitialized);*/
/*    assert(m.getDev());*/
/*    assert(m.n() % 2 == 0);*/
/*    kAddGaussianNoise<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(sdRndMults, sdRndWords, m.getDev(),stddev,m.n());*/
/*    cutilSafeCall(cudaThreadSynchronize());*/
/*    checkCudaError("Kernel execution failed");*/
/*}*/

/*template<class T>*/
/*__global__ void kBinarizeProbs(unsigned int* rndMults, unsigned long long* rndWords, T *gData, unsigned int numElements) {*/
/*    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;*/
/*    unsigned long long rndWord = rndWords[idx];*/
/*    const unsigned int rndMult = rndMults[idx];*/

/*    for(unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {*/
/*        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);*/
/*        gData[i] = gData[i] > (__uint2float_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;*/
/*    }*/
/*    rndWords[idx] = rndWord;*/
/*}*/

/*template<class T>*/
/*void MatrixTools<T>::binarize_probs(T& m) {*/
/*    assert(sRndInitialized);*/
/*    assert(m.getDev());*/
/*    kBinarizeProbs<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(sdRndMults, sdRndWords, m.getDev(),m.n());*/
/*    cutilSafeCall(cudaThreadSynchronize());*/
/*    checkCudaError("Kernel execution failed");*/
/*}*/


// New RNG
 
#define MT_MM     9
#define MT_NN     19
#define MT_WMASK  0xFFFFFFFFU
#define MT_UMASK  0xFFFFFFFEU
#define MT_LMASK  0x1U
#define MT_RNG_COUNT 32768
#define MT_SHIFT0 12
#define MT_SHIFTB 7
#define MT_SHIFTC 15
#define MT_SHIFT1 18
#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)
 
// Record format for MersenneTwister.dat, created by spawnTwisters.c
struct mt_struct_stripped {
	unsigned int matrix_a;
	unsigned int mask_b;
	unsigned int mask_c;
	unsigned int seed;
};
 
// Per-thread state object for a single twister.
struct MersenneTwisterState {
	unsigned int mt[MT_NN];
	int iState;
	unsigned int mti1;
};
 
// Preloaded, offline-generated seed data structure.
__device__ static mt_struct_stripped MT[MT_RNG_COUNT];
__device__ static MersenneTwisterState gStates[MT_RNG_COUNT];
 
__device__ unsigned int MersenneTwisterGenerate(MersenneTwisterState &state, unsigned int threadID) {
	int iState1 = state.iState + 1;
	int iStateM = state.iState + MT_MM;
 
	if(iState1 >= MT_NN) iState1 -= MT_NN;
	if(iStateM >= MT_NN) iStateM -= MT_NN;
 
	unsigned int mti = state.mti1;
	state.mti1 = state.mt[iState1];
	unsigned int mtiM = state.mt[iStateM];
 
	unsigned int x = (mti & MT_UMASK) | (state.mti1 & MT_LMASK);
	x = mtiM ^ (x >> 1) ^ ((x & 1) ? MT[threadID].matrix_a : 0);
	state.mt[state.iState] = x;
	state.iState = iState1;
 
	// Tempering transformation.
	x ^= (x >> MT_SHIFT0);
	x ^= (x << MT_SHIFTB) & MT[threadID].mask_b;
	x ^= (x << MT_SHIFTC) & MT[threadID].mask_c;
	x ^= (x >> MT_SHIFT1);
 
	return x;
}
 
#define TWISTER_WARM_UP 0
 
__device__ void MersenneTwisterInitialize(MersenneTwisterState &state, unsigned int threadID) {
	state.mt[0] = MT[threadID].seed;
	for(int i = 1; i < MT_NN; ++ i) {
		state.mt[i] = (1812433253U * (state.mt[i - 1] ^ (state.mt[i - 1] >> 30)) + i) & MT_WMASK;
	}
 
	state.iState = 0;
	state.mti1 = state.mt[0];
 
	// warm up the twister
        #if TWISTER_WARM_UP
	for(int i = 0; i < 10000; ++ i) {
		MersenneTwisterGenerate(state, threadID);
	}
        #endif
 
}
 
 
//Box Muller transform
#define PI 3.14159265358979f
__device__ 
void BoxMuller(float &u1, float &u2){
    float   r = sqrtf(-2.0f * logf(u1)); 
    float phi = 2 * PI * u2;
 
    u1 = (r * __cosf(phi));
	u2 = (r * __sinf(phi));
}
 
template<class value_type>
struct binarize{
	__device__
	void operator()(value_type* dst, const int& n) const {
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if( idx >= n ) return;
		/*__shared__ MersenneTwisterState mtState;*/
		MersenneTwisterState mtState = gStates[idx];
		for(int i=idx; i<n; i += blockDim.x * gridDim.x)
			 dst[i] = ((value_type(MersenneTwisterGenerate(mtState, idx)) / 4294967295.0f) < dst[i]);
		gStates[idx] = mtState;
	}
};

template<class value_type>
struct rnd_uniform{
	const value_type m_vmin;
	const value_type m_vmax;

	rnd_uniform(const value_type& vmin, const value_type& vmax) : 
		m_vmin(vmin),
		m_vmax(vmax) { }
 
	__device__
	void operator()(value_type* dst, const int& n) const {
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if( idx >= n ) return;
		/*__shared__ MersenneTwisterState mtState;*/
		MersenneTwisterState mtState = gStates[idx];
		for(int i=idx; i<n; i += blockDim.x * gridDim.x)
			 dst[i] = value_type(MersenneTwisterGenerate(mtState, idx)) / 4294967295.0f;
		gStates[idx] = mtState;
	}
};

__global__
void rnd_init_dev() {
	unsigned int idx = __mul24(blockIdx.x , blockDim.x) + threadIdx.x;
	/*__shared__ MersenneTwisterState mtState;*/
	MersenneTwisterState mtState = gStates[idx];
	MersenneTwisterInitialize(mtState, idx);
	gStates[idx] = mtState;
}

template<class value_type>
struct rnd_normal {
	rnd_normal() { }
 
	__device__
		void 
	operator()(value_type* dst, const int& n) const {
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if( idx >= n ) return;
		/*__shared__ MersenneTwisterState mtState;*/
		MersenneTwisterState mtState = gStates[idx];
		float x,y;
		for(unsigned int i=idx; i<n-1; i += blockDim.x * gridDim.x){
			 float2 tmp=dst[i]; // move up so it can be done in background while we fetch random numbers
			 do{
				 x = float(MersenneTwisterGenerate(mtState, idx)) / 4294967295.0f;
				 y = float(MersenneTwisterGenerate(mtState, idx)) / 4294967295.0f;
				 BoxMuller(x, y); //transform uniform into two independent standard normals
			 }while(!isfinite(x) || !isfinite(y));

			 dst[i] = make_float2(x+tmp.x,y+tmp.y);
		}
		__syncthreads();
		gStates[idx] = mtState;
	}
};

namespace cuv{
	// Initialize seeds for the Mersenne Twister
	void initialize_mersenne_twister_seeds() {
		mt_struct_stripped *mtStripped = new mt_struct_stripped[MT_RNG_COUNT];
		FILE *datFile = fopen((std::string(QUOTEME(RANDOM_PATH))+"/MersenneTwister.dat").c_str(), "rb");
		if(!datFile){
			cuvAssert(datFile);
		}
		bool ret = fread(mtStripped, sizeof(mt_struct_stripped) * MT_RNG_COUNT, 1, datFile);
		assert(ret);
		fclose(datFile);

		// Seed the structure with low-quality random numbers. Twisters will need "warming up"
		// before the RNG quality improves.
		srand(time(0));
		for(int i = 0; i < MT_RNG_COUNT; ++ i) {
			mtStripped[i].seed = rand();
		}

		// Upload the initial configurations to the GPU.
		cuvSafeCall(cudaMemcpyToSymbol(MT, mtStripped, sizeof(mt_struct_stripped) * MT_RNG_COUNT, 0, cudaMemcpyHostToDevice));
		dim3 threads(512,1);
		dim3 grid(MT_RNG_COUNT/512,1,1);
		rnd_init_dev<<<grid,threads>>>();

		cuvSafeCall(cudaThreadSynchronize());
		delete[] mtStripped;
	}

	__global__ void kBinarize  (float* dst,int n, binarize<float> rng)    { rng(dst,n); }
	__global__ void kRndUniform(float* dst, int n, rnd_uniform<float> rng){ rng(dst,n); }
	__global__ void kRndNormal (float2* dst,int n, rnd_normal<float2> rng){ rng(dst,n); }

	template<>
	void rnd_binarize(dev_vector<float>& v){
		cuvAssert(v.ptr());

		binarize<float> rng;
		dim3 threads(512,1);
		dim3 grid(MT_RNG_COUNT/512,1,1);
		kBinarize<<<grid,threads>>>(v.ptr(),v.size(),rng);
		cuvSafeCall(cudaThreadSynchronize());
	}
	template<>
	void rnd_binarize(host_vector<float>& v){
	   cuvAssert(v.ptr());
	   host_vector<float>::value_type* ptr = v.ptr();
	   for(int i=0;i<v.size();i++)
		   *ptr++ = ((float)rand()/RAND_MAX) < *ptr;
	}
	template<>
	void fill_rnd_uniform(host_vector<float>& v){
	   cuvAssert(v.ptr());
	   host_vector<float>::value_type* ptr = v.ptr();
	   for(int i=0;i<v.size();i++)
		   *ptr++ = ((float)rand()/RAND_MAX);
	}
	template<>
	void fill_rnd_uniform(dev_vector<float>& v){
		cuvAssert(v.ptr());

		rnd_uniform<float> rng(0.f,1.f);
		dim3 threads(512,1);
		dim3 grid(MT_RNG_COUNT/512,1,1);
		kRndUniform<<<grid,threads>>>(v.ptr(),v.size(),rng);

		cuvSafeCall(cudaThreadSynchronize());
	}
	template<>
	void add_rnd_normal(host_vector<float>& v){
	   cuvAssert(v.ptr());
	   host_vector<float>::value_type* ptr = v.ptr();
	   typedef boost::mt19937 rng_type;
	   rng_type rng;
	   boost::normal_distribution<float> nd;
	   boost::variate_generator<rng_type, boost::normal_distribution<float> > die(rng, nd);
	   for(int i=0;i<v.size();i++)
		   *ptr++ += die();
	}
	template<>
	void add_rnd_normal(dev_vector<float>& v){
		cuvAssert(v.ptr());
		cuvAssert((v.size()%2) == 0);
		rnd_normal<float2> rng;
		dim3 threads(512,1);
		dim3 grid(MT_RNG_COUNT/512,1,1);
		using namespace std;
		kRndNormal<<<grid,threads>>>((float2*)v.ptr(),v.size()/2,rng);
		cuvSafeCall(cudaThreadSynchronize());
	}
} // cuv
