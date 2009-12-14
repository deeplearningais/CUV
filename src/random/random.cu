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
 
// 
__global__ void TestMersenneTwister(float *outArr, int nNumbers) {
	unsigned int tid = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
 
	MersenneTwisterState mtState;
	MersenneTwisterInitialize(mtState, tid);
 
	for(int i = tid; i < nNumbers; i += __mul24(blockDim.x, gridDim.x)) {
		// Make a floating-point number between 0...1 from integer 0...UINT_MAX.
		outArr[i] = float(MersenneTwisterGenerate(mtState, tid)) / 4294967295.0f;
	}
}
 
 
// Generates random numbers on the GPU
void GenerateRandomNumbers(float *randomNumbers, int nRandomNumbers) {
	// Read offline-generated initial configuration file.
	float *randomNumbersDev;
	cuvSafeCall(cudaMalloc((void **)&randomNumbersDev, sizeof(float) * nRandomNumbers));
 
	dim3 threads(512, 1);
	dim3 grid(MT_RNG_COUNT / 512, 1, 1);
 
	TestMersenneTwister<<<grid, threads>>>(randomNumbersDev, nRandomNumbers);
 
	cuvSafeCall(cudaMemcpy(randomNumbers, randomNumbersDev, sizeof(float) * nRandomNumbers, cudaMemcpyDeviceToHost));
	cuvSafeCall(cudaFree(randomNumbersDev));
}
// End Mersenne Twister Code
 
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
struct rnd_uniform{
	const value_type m_vmin;
	const value_type m_vmax;

	rnd_uniform(const value_type& vmin, const value_type& vmax) : 
		m_vmin(vmin),
		m_vmax(vmax) { }
 
	__device__
	value_type operator()() const {
		__shared__ MersenneTwisterState mtState;
		unsigned int thrOffset = blockIdx.x * blockDim.x + threadIdx.x;
		MersenneTwisterInitialize(mtState, thrOffset);
		return value_type(MersenneTwisterGenerate(mtState, thrOffset)) / 4294967295.0f;
	}
};

template<class value_type>
struct rnd_normal {
	rnd_normal() { }
 
	__device__
	value_type operator()() const {
		__shared__ MersenneTwisterState mtState;
		unsigned int thrOffset = blockIdx.x * blockDim.x + threadIdx.x;
		MersenneTwisterInitialize(mtState, thrOffset);
		float u1 = float(MersenneTwisterGenerate(mtState, thrOffset)) / 4294967295.0f;
		float u2 = float(MersenneTwisterGenerate(mtState, thrOffset)) / 4294967295.0f;
		BoxMuller(u1, u2); //transform uniform into two independent standard normals
		/*u1 = u1 * __expf( sigma); oder so*/
		/*u2 = u2 * __expf( sigma); oder so*/
		return u1; // TODO: this is SLOWWW
	}
};

namespace cuv{
	// Initialize seeds for the Mersenne Twister
	void initialize_mersenne_twister_seeds() {
		mt_struct_stripped *mtStripped = new mt_struct_stripped[MT_RNG_COUNT];

		FILE *datFile = fopen("MersenneTwister.dat", "rb");
		assert(datFile);
		assert(fread(mtStripped, sizeof(mt_struct_stripped) * MT_RNG_COUNT, 1, datFile));
		fclose(datFile);

		// Seed the structure with low-quality random numbers. Twisters will need "warming up"
		// before the RNG quality improves.
		srand(time(0));
		for(int i = 0; i < MT_RNG_COUNT; ++ i) {
			mtStripped[i].seed = rand();
		}

		// Upload the initial configurations to the GPU.
		cuvSafeCall(cudaMemcpyToSymbol(MT, mtStripped, sizeof(mt_struct_stripped) * MT_RNG_COUNT, 0, cudaMemcpyHostToDevice));
		delete[] mtStripped;
	}

	void fill_rnd_uniform(dev_vector<float>& v){
		thrust::device_ptr<float> p(v.ptr());
		thrust::generate(p, p+v.size(), rnd_uniform<float>(0.f,1.f));
	}
	void fill_rnd_normal(dev_vector<float>& v){
		thrust::device_ptr<float> p(v.ptr());
		thrust::generate(p, p+v.size(), rnd_normal<float>());
	}
} // cuv
