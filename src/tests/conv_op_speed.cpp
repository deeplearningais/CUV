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

#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/dense_matrix.hpp>
#include <cuv/vector_ops/vector_ops.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>
#include <cuv/tools/timing.hpp>
#include <cuv/random/random.hpp>
#include <cuv/matrix_ops/rprop.hpp>
#include <cuv/basics/filter_factory.hpp>
#include <cuv/convert/convert.hpp>

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

	dense_matrix<float, row_major, dev_memory_space>  img_dev(c, n*n);
	dense_matrix<float, row_major, dev_memory_space>  filter_dev(f, g*g);
	dense_matrix<float, row_major, dev_memory_space>  dst_dev(c, f*k*k);

	dense_matrix<float, row_major, host_memory_space>  img_host(c, n*n);
	dense_matrix<float, row_major, host_memory_space>  filter_host(f, g*g);
	dense_matrix<float, row_major, host_memory_space>  dst_host(c, f*k*k);

	fill(dst_dev, 0.0f);
	sequence(img_dev);    apply_scalar_functor(img_dev,   SF_MULT,0.001f);
	sequence(filter_dev); apply_scalar_functor(filter_dev,SF_MULT,0.001f);

	fill(dst_host, 1.0f);
	sequence(img_host);    apply_scalar_functor(img_host,   SF_MULT,0.001f);
	sequence(filter_host); apply_scalar_functor(filter_host,SF_MULT,0.001f);

	MEASURE_TIME(dev,  convolve(dst_dev,img_dev,filter_dev), 10);
	MEASURE_TIME(host, convolve(dst_host,img_host,filter_host), 10);

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

	dense_matrix<float, row_major, dev_memory_space>  img_dev(c, n*n);
	dense_matrix<float, row_major, dev_memory_space>  filter_dev(c, f*g*g);
	dense_matrix<float, row_major, dev_memory_space>  dst_dev(c, k*k);

	dense_matrix<float, row_major, host_memory_space>  img_host(c, n*n);
	dense_matrix<float, row_major, host_memory_space>  filter_host(c, f*g*g);
	dense_matrix<float, row_major, host_memory_space>  dst_host(c, k*k);

	fill(dst_dev, 0.0f);
	sequence(img_dev);    apply_scalar_functor(img_dev,   SF_MULT,0.001f);
	sequence(filter_dev); apply_scalar_functor(filter_dev,SF_MULT,0.001f);

	fill(dst_host, 1.0f);
	sequence(img_host);    apply_scalar_functor(img_host,   SF_MULT,0.001f);
	sequence(filter_host); apply_scalar_functor(filter_host,SF_MULT,0.001f);



	MEASURE_TIME(dev,  conv_rlcnp<dev_memory_space>(dst_dev,img_dev,filter_dev,c,n,f,g), 10);
	MEASURE_TIME(host, conv_rlcnp<host_memory_space>(dst_host,img_host,filter_host,c,n,f,g), 10);
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

	dense_matrix<float, row_major, dev_memory_space>  img_dev(c, n*n);
	dense_matrix<float, row_major, dev_memory_space>  filter_dev(c, f*g*g);
	dense_matrix<float, row_major, dev_memory_space>  dst_dev(c, f*k*k);

	dense_matrix<float, row_major, host_memory_space>  img_host(c, n*n);
	dense_matrix<float, row_major, host_memory_space>  filter_host(c, f*g*g);
	dense_matrix<float, row_major, host_memory_space>  dst_host(c, f*k*k);

	fill(dst_dev, 0.0f);
	sequence(img_dev);    apply_scalar_functor(img_dev,   SF_MULT,0.001f);
	sequence(filter_dev); apply_scalar_functor(filter_dev,SF_MULT,0.001f);

	fill(dst_host, 1.0f);
	sequence(img_host);    apply_scalar_functor(img_host,   SF_MULT,0.001f);
	sequence(filter_host); apply_scalar_functor(filter_host,SF_MULT,0.001f);

	MEASURE_TIME(dev,  convolve2(dst_dev,img_dev,filter_dev, f), 10);
	MEASURE_TIME(host, convolve2(dst_host,img_host,filter_host, f), 10);

	printf("Speedup: %3.4f\n", host/dev);
}


void conv1_iter(dense_matrix<float, row_major, dev_memory_space>& img_dev,
		dense_matrix<float, row_major, dev_memory_space>& filter_dev,
		dense_matrix<float, row_major, dev_memory_space>& dst_dev,
		int numInputMaps) {
	fill(dst_dev, 0.0f);
	for(int i = 0; i < numInputMaps; i++)
		convolve(dst_dev, img_dev, filter_dev);
}

void conv1_pass(dense_matrix<float, row_major, dev_memory_space>& img_dev,
		dense_matrix<float, row_major, dev_memory_space>& filter_dev,
		dense_matrix<float, row_major, dev_memory_space>& dst_dev,
		dense_matrix<float, row_major, dev_memory_space>& temp_dev,
		int numInputMaps) {
	fill(dst_dev, 0.0f);
	int numFilters = dst_dev.h();
	convolve(temp_dev, img_dev, filter_dev, numInputMaps);
	temp_dev.resize(temp_dev.h()/numFilters, temp_dev.w()*numFilters);
	reduce_to_row(dst_dev.vec(), temp_dev);
	temp_dev.resize(temp_dev.h()*numFilters, temp_dev.w()/numFilters);
}

void conv2_pass(dense_matrix<float, row_major, dev_memory_space>& img_dev,
		dense_matrix<float, row_major, dev_memory_space>& filter_dev,
		dense_matrix<float, row_major, dev_memory_space>& dst_dev,
		dense_matrix<float, row_major, dev_memory_space>& temp_dev,
		int numPatterns) {
	fill(temp_dev, 0.0f);
	convolve2(temp_dev, img_dev, filter_dev, filter_dev.h());
	// note: this is not the correct summation! but it's at least as computationally intense
	temp_dev.resize(temp_dev.h()/numPatterns, temp_dev.w()*numPatterns);
	reduce_to_row(dst_dev.vec(), temp_dev);
	temp_dev.resize(temp_dev.h()*numPatterns, temp_dev.w()/numPatterns);
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
	printf("Result are %i*%i images of size %ix%i.\n", f,f, k,k);

	dense_matrix<float, row_major, dev_memory_space>  img1_dev(p, n*n);
	dense_matrix<float, row_major, dev_memory_space>  filter1_dev(f, g*g);
	dense_matrix<float, row_major, dev_memory_space>  dst1_dev(f, p*m*m);

	sequence(img1_dev);    apply_scalar_functor(img1_dev,   SF_MULT,0.001f);
	sequence(filter1_dev); apply_scalar_functor(filter1_dev,SF_MULT,0.001f);

	dense_matrix<float, row_major, dev_memory_space>  img2_dev(k*p, n*n);
	dense_matrix<float, row_major, dev_memory_space>  filter2_dev(f, k*p*g*g);
	dense_matrix<float, row_major, dev_memory_space>  temp_dev(k*p, f*m*m);
	dense_matrix<float, row_major, dev_memory_space>  dst2_dev(f, p*m*m);

	sequence(img2_dev);    apply_scalar_functor(img2_dev,   SF_MULT,0.001f);
	sequence(filter2_dev); apply_scalar_functor(filter2_dev,SF_MULT,0.001f);

	MEASURE_TIME(conv1, conv1_iter(img1_dev, filter1_dev, dst1_dev, k), 10);
	MEASURE_TIME(conv2, conv2_pass(img2_dev, filter2_dev, dst2_dev, temp_dev, p), 10);

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
	printf("Result are %i*%i images of size %ix%i.\n", f,f, k,k);

	dense_matrix<float, row_major, dev_memory_space>  img1_dev(p, n*n);
	dense_matrix<float, row_major, dev_memory_space>  filter1_dev(f, g*g);
	dense_matrix<float, row_major, dev_memory_space>  dst1_dev(f, p*m*m);

	sequence(img1_dev);    apply_scalar_functor(img1_dev,   SF_MULT,0.001f);
	sequence(filter1_dev); apply_scalar_functor(filter1_dev,SF_MULT,0.001f);

	dense_matrix<float, row_major, dev_memory_space>  img2_dev(k*p, n*n);
	dense_matrix<float, row_major, dev_memory_space>  filter2_dev(k*f, g*g);
	dense_matrix<float, row_major, dev_memory_space>  temp_dev(k*f, p*m*m);
	dense_matrix<float, row_major, dev_memory_space>  dst2_dev(f, p*m*m);

	sequence(img2_dev);    apply_scalar_functor(img2_dev,   SF_MULT,0.001f);
	sequence(filter2_dev); apply_scalar_functor(filter2_dev,SF_MULT,0.001f);

	MEASURE_TIME(conv_iter, conv1_iter(img1_dev, filter1_dev, dst1_dev, k), 10);
	MEASURE_TIME(conv_pass, conv1_pass(img2_dev, filter2_dev, dst2_dev, temp_dev, k), 10);

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

	dense_matrix<float,row_major, host_memory_space> img_host(c,n*n);
	dense_matrix<float,row_major, host_memory_space> img_dev(c,n*n);
	dense_matrix<float,row_major, host_memory_space> pooled_host(c,o*o);
	dense_matrix<float,row_major, host_memory_space> pooled_dev(c,o*o);
	dense_matrix<int,row_major, host_memory_space> indices_host(c,o*o);
	dense_matrix<int,row_major, host_memory_space> indices_dev(c,o*o);


	initialize_mersenne_twister_seeds();

	// part 1: calculate matrix indices
	fill_rnd_uniform(img_dev.vec());
	convert(img_host, img_dev);

	MEASURE_TIME(host_max, max_pooling(pooled_host, img_host, p, 0, &indices_host), 10 );
	MEASURE_TIME(dev_max, max_pooling(pooled_dev, img_dev, p, 0, &indices_dev), 10 );

	printf("Speedup pooling: %3.4f\n", host_max/dev_max);

	// part 2: propagate back to those indices
	fill_rnd_uniform(pooled_dev.vec());
	convert(pooled_host, pooled_dev);

	fill(img_host, 0.f);
	fill(img_dev, 0.f);

	MEASURE_TIME(host_sup, supersample(img_host, pooled_host, p, &indices_host), 10 );
	MEASURE_TIME(dev_sup, supersample(img_dev, pooled_dev, p, &indices_dev), 10 );

	printf("Speedup: %3.4f\n", host_sup/dev_sup);

}

BOOST_AUTO_TEST_CASE( max_pool_with_overlap )
{
	const int n = 65;
	int p = 9;
	int l = 5;
	const int m = (n-p)/(p-l)+1; // resulting image size
	const int c = 100;

	dense_matrix<float,row_major, host_memory_space> img_host(c,n*n);
	dense_matrix<float,row_major, host_memory_space> dst_host(c,m*m);
	dense_matrix<int,row_major, host_memory_space> indices_host(c,m*m);

	dense_matrix<float,row_major, host_memory_space> img_dev(c,n*n);
	dense_matrix<float,row_major, host_memory_space> dst_dev(c,m*m);
	dense_matrix<int,row_major, host_memory_space> indices_dev(c,m*m);

	sequence(img_host);
	sequence(img_dev);

	MEASURE_TIME(host, max_pooling(dst_host, img_host, p, l, &indices_host), 10 );
	MEASURE_TIME(dev, max_pooling(dst_dev, img_dev, p, l, &indices_dev), 10 );

	printf("Speedup forward: %3.4f\n", host/dev);

	MEASURE_TIME(host_bw, super_to_max(img_host, dst_host, p, l, &indices_host), 10 );
	MEASURE_TIME(dev_bw, super_to_max(img_dev, dst_dev, p, l, &indices_dev), 10 );

	printf("Speedup backward: %3.4f\n", host_bw/dev_bw);
}

BOOST_AUTO_TEST_SUITE_END()
