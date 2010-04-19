//*LB*
// Copyright (c) 2010, Hannes Schulz, Andreas Mueller, Dominik Scherer
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
#include <boost/test/included/unit_test.hpp>

#include <cuv_general.hpp>
#include <dense_matrix.hpp>
#include <vector_ops.hpp>
#include <matrix_ops.hpp>
#include <convolution_ops.hpp>
#include <timing.hpp>
#include <random.hpp>
#include <matrix_ops/rprop.hpp>
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

BOOST_AUTO_TEST_CASE( convolution_speed )
{
	conv_speed_test(140, 16, 16, 30);
	conv_speed_test(40, 9, 16, 128);
	conv_speed_test(47, 15, 16, 1);
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
