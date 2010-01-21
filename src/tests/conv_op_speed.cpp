#define BOOST_TEST_MODULE example
#include <cstdio>
#include <boost/test/included/unit_test.hpp>

#include <cuv_general.hpp>
#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>
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

void conv_speed_test(int inputSize, int filterSize, int numFilters, int numImages)
{
	int c = numImages;
	int n = inputSize;
	int f = numFilters;
	int g = filterSize;
	int k = inputSize-filterSize+1;

	printf("Convolving %i images of size %ix%i with %i filters of size %ix%i.\n", c, n,n, f, g,g);
	printf("Result are %i*%i images of size %ix%i\n", c,f, k,k);

	dev_dense_matrix<float, row_major>  d_img(c, n*n);
	dev_dense_matrix<float, row_major>  d_filter(f, g*g);
	dev_dense_matrix<float, row_major>  d_dst(c, f*k*k);

	host_dense_matrix<float, row_major>  h_img(c, n*n);
	host_dense_matrix<float, row_major>  h_filter(f, g*g);
	host_dense_matrix<float, row_major>  h_dst(c, f*k*k);

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

	host_dense_matrix<float, row_major> A(imgs,size*size);
	host_dense_matrix<float, row_major> B(imgs,size*size*factor*factor);
	dev_dense_matrix<float, row_major> dev_A(imgs,size*size);
	dev_dense_matrix<float, row_major> dev_B(imgs,size*size*factor*factor);

	sequence(A);
	sequence(dev_A);

	MEASURE_TIME(host, supersample(B, A, factor), 10);
	MEASURE_TIME(dev,  supersample(dev_B, dev_A, factor), 10);
	printf("Speedup: %3.4f\n", host/dev);
}

BOOST_AUTO_TEST_SUITE_END()
