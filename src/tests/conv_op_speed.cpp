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





#define BOOST_TEST_MODULE example
#include <cstdio>
#include <memory>
#include <boost/test/included/unit_test.hpp>

#include <cuv_general.hpp>
#include <dense_matrix.hpp>
#include <vector_ops.hpp>
#include <matrix_ops.hpp>
#include <convolution_ops.hpp>
#include <timing.hpp>
#include <random.hpp>
#include <matrix_ops/rprop.hpp>
#include <basics/filter_factory.hpp>
#include <convert.hpp>

using namespace cuv;

#define MEASURE_TIME(MSG, OPERATION, ITERS)     \
	float MSG;                                  \
	if(1){                                      \
		Timing tim;                             \
		for(int i=0;i<ITERS;i++){               \
			OPERATION ;                         \
		}                                       \
		tim.update(ITERS);                      \
		printf("%s [%s] took %4.4f us/pass\n", #MSG, #OPERATION, 1000000.0f*tim.perf()); \
		MSG = 1000000.0f*tim.perf();            \
	}

struct MyConfig {
	static const int dev = CUDA_TEST_DEVICE;
	MyConfig()   { 
		printf("Testing on device=%d\n",dev);
		initCUDA(dev); 
	}
	~MyConfig()  { exitCUDA();  }
};


BOOST_GLOBAL_FIXTURE( MyConfig );


struct Fix{
	Fix()
	{
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )

BOOST_AUTO_TEST_CASE( reorder_speed )
{
	int size = 4*4;
	int x = 10;
	int y = 16;

	dense_matrix<float, row_major, host_memory_space> A(y,x*size);
	dense_matrix<float, row_major, dev_memory_space> dev_A(y,x*size);

	sequence(A);
	sequence(dev_A);

	MEASURE_TIME(host, reorder(A, size), 1);
	MEASURE_TIME(dev,  reorder(dev_A, size), 1);
	printf("Speedup: %3.4f\n", host/dev);
}

void conv_speed_test(int inputSize, int filterSize, int numFilters, int numImages)
{
	int c = numImages;
	int n = inputSize;
	int f = numFilters;
	int g = filterSize;
	int k = inputSize-filterSize+1;

	printf("Convolving %i images of size %ix%i with %i filters of size %ix%i.\n", c, n,n, f, g,g);
	printf("Result are %i*%i images of size %ix%i\n", c,f, k,k);

	dense_matrix<float, row_major, dev_memory_space>  d_img(c, n*n);
	dense_matrix<float, row_major, dev_memory_space>  d_filter(f, g*g);
	dense_matrix<float, row_major, dev_memory_space>  d_dst(c, f*k*k);

	dense_matrix<float, row_major, host_memory_space>  h_img(c, n*n);
	dense_matrix<float, row_major, host_memory_space>  h_filter(f, g*g);
	dense_matrix<float, row_major, host_memory_space>  h_dst(c, f*k*k);

	fill(d_dst, 0.0f);
	sequence(d_img);    apply_scalar_functor(d_img,   SF_MULT,0.001f);
	sequence(d_filter); apply_scalar_functor(d_filter,SF_MULT,0.001f);

	fill(h_dst, 1.0f);
	sequence(h_img);    apply_scalar_functor(h_img,   SF_MULT,0.001f);
	sequence(h_filter); apply_scalar_functor(h_filter,SF_MULT,0.001f);

	MEASURE_TIME(dev,  convolve(d_dst,d_img,d_filter), 10);
	MEASURE_TIME(host, convolve(h_dst,h_img,h_filter), 10);

	printf("Speedup: %3.4f\n", host/dev);
}


template<class ms_type>
void conv_rlcnp(dense_matrix<float, row_major, ms_type>& dst,
				dense_matrix<float, row_major, ms_type>& img,
				dense_matrix<float, row_major, ms_type>& filter,
				int numImages,
				int inputSize,
				int numFilter,
				int filterSize
				){
	int c = inputSize;
	int k = inputSize - filterSize + 1;
	int f = numFilter;
	dense_matrix<float, row_major, ms_type>  helper(c, f*k*k);

	convolve2(helper, img, filter, numFilter);
	reduce_to_row(dst.vec(), helper, RF_ADD);

}

void conv_rlcnp_test(int inputSize, int filterSize, int numFilters, int numImages){

	int c = numImages;
	int n = inputSize;
	int f = numFilters;
	int g = filterSize;
	int k = inputSize-filterSize+1;

	printf("Convolving %i images of size %ix%i each with %i filters of size %ix%i using convolve2().\n", c, n,n, f, g,g);
	printf("Result are %i*%i images of size %ix%i\n", c,f, k,k);

	dense_matrix<float, row_major, dev_memory_space>  d_img(c, n*n);
	dense_matrix<float, row_major, dev_memory_space>  d_filter(c, f*g*g);
	dense_matrix<float, row_major, dev_memory_space>  d_dst(c, k*k);

	dense_matrix<float, row_major, host_memory_space>  h_img(c, n*n);
	dense_matrix<float, row_major, host_memory_space>  h_filter(c, f*g*g);
	dense_matrix<float, row_major, host_memory_space>  h_dst(c, k*k);

	fill(d_dst, 0.0f);
	sequence(d_img);    apply_scalar_functor(d_img,   SF_MULT,0.001f);
	sequence(d_filter); apply_scalar_functor(d_filter,SF_MULT,0.001f);

	fill(h_dst, 1.0f);
	sequence(h_img);    apply_scalar_functor(h_img,   SF_MULT,0.001f);
	sequence(h_filter); apply_scalar_functor(h_filter,SF_MULT,0.001f);



	MEASURE_TIME(dev,  conv_rlcnp<dev_memory_space>(d_dst,d_img,d_filter,c,n,f,g), 10);
	MEASURE_TIME(host, conv_rlcnp<host_memory_space>(h_dst,h_img,h_filter,c,n,f,g), 10);
	printf("Speedup: %3.4f\n", host/dev);

}


void conv2_speed_test(int inputSize, int filterSize, int numFilters, int numImages)
{
	int c = numImages;
	int n = inputSize;
	int f = numFilters;
	int g = filterSize;
	int k = inputSize-filterSize+1;

	printf("Convolving %i images of size %ix%i each with %i filters of size %ix%i using convolve2().\n", c, n,n, f, g,g);
	printf("Result are %i*%i images of size %ix%i\n", c,f, k,k);

	dense_matrix<float, row_major, dev_memory_space>  d_img(c, n*n);
	dense_matrix<float, row_major, dev_memory_space>  d_filter(c, f*g*g);
	dense_matrix<float, row_major, dev_memory_space>  d_dst(c, f*k*k);

	dense_matrix<float, row_major, host_memory_space>  h_img(c, n*n);
	dense_matrix<float, row_major, host_memory_space>  h_filter(c, f*g*g);
	dense_matrix<float, row_major, host_memory_space>  h_dst(c, f*k*k);

	fill(d_dst, 0.0f);
	sequence(d_img);    apply_scalar_functor(d_img,   SF_MULT,0.001f);
	sequence(d_filter); apply_scalar_functor(d_filter,SF_MULT,0.001f);

	fill(h_dst, 1.0f);
	sequence(h_img);    apply_scalar_functor(h_img,   SF_MULT,0.001f);
	sequence(h_filter); apply_scalar_functor(h_filter,SF_MULT,0.001f);

	MEASURE_TIME(dev,  convolve2(d_dst,d_img,d_filter, f), 10);
	MEASURE_TIME(host, convolve2(h_dst,h_img,h_filter, f), 10);

	printf("Speedup: %3.4f\n", host/dev);
}


void conv1_iter(dense_matrix<float, row_major, dev_memory_space>& d_img,
		dense_matrix<float, row_major, dev_memory_space>& d_filter,
		dense_matrix<float, row_major, dev_memory_space>& d_dst,
		int numInputMaps) {
	fill(d_dst, 0.0f);
	for(int i = 0; i < numInputMaps; i++)
		convolve(d_dst, d_img, d_filter);
}

void conv1_pass(dense_matrix<float, row_major, dev_memory_space>& d_img,
		dense_matrix<float, row_major, dev_memory_space>& d_filter,
		dense_matrix<float, row_major, dev_memory_space>& d_dst,
		dense_matrix<float, row_major, dev_memory_space>& d_temp,
		int numInputMaps) {
	fill(d_dst, 0.0f);
	int numFilters = d_dst.h();
	convolve(d_temp, d_img, d_filter, numInputMaps);
	d_temp.resize(d_temp.h()/numFilters, d_temp.w()*numFilters);
	reduce_to_row(d_dst.vec(), d_temp);
	d_temp.resize(d_temp.h()*numFilters, d_temp.w()/numFilters);
}

void conv2_pass(dense_matrix<float, row_major, dev_memory_space>& d_img,
		dense_matrix<float, row_major, dev_memory_space>& d_filter,
		dense_matrix<float, row_major, dev_memory_space>& d_dst,
		dense_matrix<float, row_major, dev_memory_space>& d_temp,
		int numPatterns) {
	fill(d_temp, 0.0f);
	convolve2(d_temp, d_img, d_filter, d_filter.h());
	// note: this is not the correct summation! but it's at least as computationally intense
	d_temp.resize(d_temp.h()/numPatterns, d_temp.w()*numPatterns);
	reduce_to_row(d_dst.vec(), d_temp);
	d_temp.resize(d_temp.h()*numPatterns, d_temp.w()/numPatterns);
}

// compare whether iterating over conv() or calling conv2() and accumulating is faster
void conv_vs_conv2_speed(int inputSize, int numInputMaps, int filterSize, int numFilters, int numPatterns)
{
	int p = numPatterns;
	int k = numInputMaps;
	int f = numFilters; // per input map

	int n = inputSize;
	int g = filterSize;
	int m = inputSize-filterSize+1;

	printf("Convolving %i images of size %ix%i each with %i filters of size %ix%i (for %i patterns)\n", k, n,n, f, g,g, p);
	printf("Result are %i*%i images of size %ix%i.\n", f, k,k);

	dense_matrix<float, row_major, dev_memory_space>  d_img1(p, n*n);
	dense_matrix<float, row_major, dev_memory_space>  d_filter1(f, g*g);
	dense_matrix<float, row_major, dev_memory_space>  d_dst1(f, p*m*m);

	sequence(d_img1);    apply_scalar_functor(d_img1,   SF_MULT,0.001f);
	sequence(d_filter1); apply_scalar_functor(d_filter1,SF_MULT,0.001f);

	dense_matrix<float, row_major, dev_memory_space>  d_img2(k*p, n*n);
	dense_matrix<float, row_major, dev_memory_space>  d_filter2(f, k*p*g*g);
	dense_matrix<float, row_major, dev_memory_space>  d_temp(k*p, f*m*m);
	dense_matrix<float, row_major, dev_memory_space>  d_dst2(f, p*m*m);

	sequence(d_img2);    apply_scalar_functor(d_img2,   SF_MULT,0.001f);
	sequence(d_filter2); apply_scalar_functor(d_filter2,SF_MULT,0.001f);

	MEASURE_TIME(conv1, conv1_iter(d_img1, d_filter1, d_dst1, k), 10);
	MEASURE_TIME(conv2, conv2_pass(d_img2, d_filter2, d_dst2, d_temp, p), 10);

	printf("Convolve1: %3.4f\n", conv1);
	printf("Convolve2: %3.4f\n", conv2);
	if(conv2<conv1)
		printf("conv2 is %3.4f times faster than conv1\n", conv1/conv2);
	else
		printf("conv1 is %3.4f times faster than conv2\n", conv2/conv1);
}

// compare whether iterating over conv() or calling conv1() and accumulating is faster
void conv_vs_conv_speed(int inputSize, int numInputMaps, int filterSize, int numFilters, int numPatterns)
{
	int p = numPatterns;
	int k = numInputMaps;
	int f = numFilters; // per input map

	int n = inputSize;
	int g = filterSize;
	int m = inputSize-filterSize+1;

	printf("Convolving %i images of size %ix%i each with %i filters of size %ix%i (for %i patterns)\n", k, n,n, f, g,g, p);
	printf("Result are %i*%i images of size %ix%i.\n", f, k,k);

	dense_matrix<float, row_major, dev_memory_space>  d_img1(p, n*n);
	dense_matrix<float, row_major, dev_memory_space>  d_filter1(f, g*g);
	dense_matrix<float, row_major, dev_memory_space>  d_dst1(f, p*m*m);

	sequence(d_img1);    apply_scalar_functor(d_img1,   SF_MULT,0.001f);
	sequence(d_filter1); apply_scalar_functor(d_filter1,SF_MULT,0.001f);

	dense_matrix<float, row_major, dev_memory_space>  d_img2(k*p, n*n);
	dense_matrix<float, row_major, dev_memory_space>  d_filter2(k*f, g*g);
	dense_matrix<float, row_major, dev_memory_space>  d_temp(k*f, p*m*m);
	dense_matrix<float, row_major, dev_memory_space>  d_dst2(f, p*m*m);

	sequence(d_img2);    apply_scalar_functor(d_img2,   SF_MULT,0.001f);
	sequence(d_filter2); apply_scalar_functor(d_filter2,SF_MULT,0.001f);

	MEASURE_TIME(conv_iter, conv1_iter(d_img1, d_filter1, d_dst1, k), 10);
	MEASURE_TIME(conv_pass, conv1_pass(d_img2, d_filter2, d_dst2, d_temp, k), 10);

	printf("conv_iter: %3.4f\n", conv_iter);
	printf("conv_pass: %3.4f\n", conv_pass);
	if(conv_pass<conv_iter)
		printf("conv_pass is %3.4f times faster than conv_iter\n", conv_iter/conv_pass);
	else
		printf("conv_iter is %3.4f times faster than conv_pass\n", conv_pass/conv_iter);
}


BOOST_AUTO_TEST_CASE( convolution_speed )
{
	//conv_speed_test(140, 16, 16, 30);
	//conv_speed_test(40, 9, 16, 128);
	//conv_speed_test(47, 15, 16, 1);
	//conv2_speed_test(384,9,32,16);
	//conv_rlcnp_test(384,9,16,16);
	//conv_rlcnp_test(180+4*2,9,16,16);

//	conv_vs_conv2_speed(382, 16, 9, 32, 1);
//	conv_vs_conv2_speed(96, 2, 5, 16, 60);

	conv_vs_conv_speed(96, 2, 5, 16, 60);
}

BOOST_AUTO_TEST_CASE( supersampling_speed )
{
	int size = 32;
	int imgs = 20;
	int factor = 8;

	dense_matrix<float, row_major, host_memory_space> A(imgs,size*size);
	dense_matrix<float, row_major, host_memory_space> B(imgs,size*size*factor*factor);
	dense_matrix<float, row_major, dev_memory_space> dev_A(imgs,size*size);
	dense_matrix<float, row_major, dev_memory_space> dev_B(imgs,size*size*factor*factor);

	sequence(A);
	sequence(dev_A);

	MEASURE_TIME(host, supersample(B, A, factor), 10);
	MEASURE_TIME(dev,  supersample(dev_B, dev_A, factor), 10);
	printf("Speedup: %3.4f\n", host/dev);
}

BOOST_AUTO_TEST_CASE( maxima_plus_index_speed )
{
	const int c = 30;
	const int n = 64;
	const int p = 4;
	const int o = n/p;

	dense_matrix<float,row_major, host_memory_space> h_img(c,n*n);
	dense_matrix<float,row_major, host_memory_space> d_img(c,n*n);
	dense_matrix<float,row_major, host_memory_space> h_pooled(c,o*o);
	dense_matrix<float,row_major, host_memory_space> d_pooled(c,o*o);
	dense_matrix<int,row_major, host_memory_space> h_indices(c,o*o);
	dense_matrix<int,row_major, host_memory_space> d_indices(c,o*o);


	initialize_mersenne_twister_seeds();

	// part 1: calculate matrix indices
	fill_rnd_uniform(d_img.vec());
	convert(h_img, d_img);

	MEASURE_TIME(host_max, max_pooling(h_pooled, h_img, p, 0, &h_indices), 10 );
	MEASURE_TIME(dev_max, max_pooling(d_pooled, d_img, p, 0, &d_indices), 10 );

	printf("Speedup pooling: %3.4f\n", host_max/dev_max);

	// part 2: propagate back to those indices
	fill_rnd_uniform(d_pooled.vec());
	convert(h_pooled, d_pooled);

	fill(h_img, 0.f);
	fill(d_img, 0.f);

	MEASURE_TIME(host_sup, supersample(h_img, h_pooled, p, &h_indices), 10 );
	MEASURE_TIME(dev_sup, supersample(d_img, d_pooled, p, &d_indices), 10 );

	printf("Speedup: %3.4f\n", host_sup/dev_sup);

}

BOOST_AUTO_TEST_CASE( max_pool_with_overlap )
{
	const int n = 65;
	int p = 9;
	int l = 5;
	const int m = (n-p)/(p-l)+1; // resulting image size
	const int c = 100;

	dense_matrix<float,row_major, host_memory_space> h_img(c,n*n);
	dense_matrix<float,row_major, host_memory_space> h_dst(c,m*m);
	dense_matrix<int,row_major, host_memory_space> h_indices(c,m*m);

	dense_matrix<float,row_major, host_memory_space> d_img(c,n*n);
	dense_matrix<float,row_major, host_memory_space> d_dst(c,m*m);
	dense_matrix<int,row_major, host_memory_space> d_indices(c,m*m);

	sequence(h_img);
	sequence(d_img);

	MEASURE_TIME(host, max_pooling(h_dst, h_img, p, l, &h_indices), 10 );
	MEASURE_TIME(dev, max_pooling(d_dst, d_img, p, l, &d_indices), 10 );

	printf("Speedup forward: %3.4f\n", host/dev);

	MEASURE_TIME(host_bw, super_to_max(h_img, h_dst, p, l, &h_indices), 10 );
	MEASURE_TIME(dev_bw, super_to_max(d_img, d_dst, p, l, &d_indices), 10 );

	printf("Speedup backward: %3.4f\n", host_bw/dev_bw);
}

BOOST_AUTO_TEST_SUITE_END()
