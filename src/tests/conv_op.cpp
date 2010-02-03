#define BOOST_TEST_MODULE example
#include <cstdio>
#include <boost/test/included/unit_test.hpp>

#include <cuv_test.hpp>
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
#include <../random/random.hpp>

using namespace cuv;

struct Fix{
	static const int c = 2;  // # patterns (images)
	static const int n = 64;  // image size
	static const int f = 16;   // # filters
	static const int p = 8;	   // pooling size
	static const int g = 8;    // filter size
	static const int k = n-g+1;// target image size
	static const int o = n/p;  // pooling output size
	dev_dense_matrix<float, row_major>  d_img,d_filter,d_dst,d_pooled;
	host_dense_matrix<float, row_major> h_img,h_filter,h_dst,h_pooled;
	Fix()
	:   d_img(c,n*n), d_filter(f,g*g), d_dst(c,f*k*k), d_pooled(c,o*o)
	,   h_img(c,n*n), h_filter(f,g*g), h_dst(c,f*k*k), h_pooled(c,o*o)
	{
		//MEASURE_TIME("warmup", apply_scalar_functor(v, SF_EXP), 100);
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )

BOOST_AUTO_TEST_CASE( convolution )
{
	fill(d_dst, 0.0f);
	sequence(d_img);    apply_scalar_functor(d_img,   SF_MULT,0.001f);
	sequence(d_filter); apply_scalar_functor(d_filter,SF_MULT,0.001f);

	fill(h_dst, 0.0f);
	sequence(h_img);    apply_scalar_functor(h_img,   SF_MULT,0.001f);
	sequence(h_filter); apply_scalar_functor(h_filter,SF_MULT,0.001f);

	convolve(d_dst,d_img,d_filter);
	convolve(h_dst,h_img,h_filter);

	host_dense_matrix<float, row_major> dst2(d_dst.h(), d_dst.w());
	convert(dst2,d_dst);

	for(int i=0;i<d_dst.h();i++){
		for(int j=0;j<d_dst.w();j++){
			BOOST_CHECK_CLOSE( dst2(i,j), h_dst(i,j), 0.001 );
		}
	}
}

BOOST_AUTO_TEST_CASE( local_maxima )
{
	fill(d_dst, 0.0f);
	sequence(d_img);    apply_scalar_functor(d_img,   SF_MULT,0.001f);

	fill(h_dst, 0.0f);
	sequence(h_img);    apply_scalar_functor(h_img,   SF_MULT,0.001f);

	local_maximum(h_pooled, h_img, p);
	local_maximum(d_pooled, d_img, p);

	host_dense_matrix<float, row_major> pooled2(d_pooled.h(), d_pooled.w());
	convert(pooled2,d_pooled);

	for(int i=0;i<d_pooled.h();i++){
		for(int j=0;j<d_pooled.w();j++){
			BOOST_CHECK_CLOSE( pooled2(i,j), h_pooled(i,j), 0.001 );
		}
	}
}

BOOST_AUTO_TEST_CASE( supersampling )
{
	fill(d_dst, 0.0f);
	sequence(d_pooled);    apply_scalar_functor(d_pooled,   SF_MULT,0.001f);

	fill(h_pooled, 0.0f);
	sequence(h_pooled);    apply_scalar_functor(h_pooled,   SF_MULT,0.001f);

	supersample(h_img, h_pooled, p);
	supersample(d_img, d_pooled, p);

	host_dense_matrix<float, row_major> img2(d_img.h(), d_img.w());
	convert(img2, d_img);

	MAT_CMP(img2, h_img, 0.001);
}

BOOST_AUTO_TEST_CASE( reorder_matrix )
{
	sequence(d_dst); apply_scalar_functor(d_dst, SF_MULT,0.001f);
	sequence(h_dst); apply_scalar_functor(h_dst, SF_MULT,0.001f);

	reorder(d_dst, k*k);
	reorder(h_dst, k*k);

	host_dense_matrix<float, row_major> dst2(d_dst.h(), d_dst.w());
	convert(dst2, d_dst);

	MAT_CMP(dst2, h_dst, 0.1);
}

BOOST_AUTO_TEST_CASE( copy_into_matrix )
{
	const int padding = 5;
	const int size = n + 2 * padding;

	host_dense_matrix<float, row_major> h_pad(h_img.h(), size * size);
	dev_dense_matrix<float, row_major> d_pad(d_img.h(), size * size);

	sequence(d_img); apply_scalar_functor(d_img, SF_MULT,0.001f);
	sequence(h_img); apply_scalar_functor(h_img, SF_MULT,0.001f);
	sequence(d_pad);
	sequence(h_pad);

	copy_into(d_pad, d_img, padding);
	copy_into(h_pad, h_img, padding);

	MAT_CMP(h_pad, d_pad, 0.1);
}

BOOST_AUTO_TEST_CASE( local_maxima_index )
{
	initialize_mersenne_twister_seeds();

	// part 1: calculate matrix indices
	fill_rnd_uniform(d_img.vec());
	convert(h_img, d_img);

	host_dense_matrix<int,row_major> h_indices(c,o*o);
	dev_dense_matrix<int,row_major> d_indices(c,o*o);

	local_maximum(h_pooled, h_img, p, &h_indices);
	local_maximum(d_pooled, d_img, p, &d_indices);

	host_dense_matrix<int, row_major> indices2(d_indices.h(), d_indices.w());
	convert(indices2,d_indices);

	for(int i=0;i<d_indices.h();i++){
		for(int j=0;j<d_indices.w();j++){
			BOOST_CHECK_EQUAL( indices2(i,j), h_indices(i,j) );
		}
	}

	// part 2: propagate back to those indices
	fill_rnd_uniform(d_pooled.vec());
	convert(h_pooled, d_pooled);

	fill(h_img, 0.f);
	fill(d_img, 0.f);

	supersample(h_img, h_pooled, p, &h_indices);
	supersample(d_img, d_pooled, p, &d_indices);

	MAT_CMP(d_img, h_img, 0.1);
}
}
