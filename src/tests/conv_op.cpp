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
#include <boost/test/included/unit_test.hpp>
#include <float.h>

#include <cuv_test.hpp>
#include <cuv_general.hpp>
#include <dense_matrix.hpp>
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
	dense_matrix<float, row_major, dev_memory_space>  d_img,d_filter,d_dst,d_pooled;
	dense_matrix<float, row_major, host_memory_space> h_img,h_filter,h_dst,h_pooled;
	Fix()
	:   d_img(c,n*n), d_filter(f,g*g), d_dst(c,f*k*k), d_pooled(c,o*o)
	,   h_img(c,n*n), h_filter(f,g*g), h_dst(c,f*k*k), h_pooled(c,o*o)
	{
		//MEASURE_TIME("warmup", apply_scalar_functor(v, SF_EXP), 100);
	}
	~Fix(){
	}
};

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

	dense_matrix<float, row_major, host_memory_space> dst2(d_dst.h(), d_dst.w());
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

	max_pooling(h_pooled, h_img, p);
	max_pooling(d_pooled, d_img, p);

	dense_matrix<float, row_major, host_memory_space> pooled2(d_pooled.h(), d_pooled.w());
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

	dense_matrix<float, row_major,host_memory_space> img2(d_img.h(), d_img.w());
	convert(img2, d_img);

	MAT_CMP(img2, h_img, 0.001);
}

BOOST_AUTO_TEST_CASE( reorder_matrix )
{
	sequence(d_dst); apply_scalar_functor(d_dst, SF_MULT,0.001f);
	sequence(h_dst); apply_scalar_functor(h_dst, SF_MULT,0.001f);

	reorder(d_dst, k*k);
	reorder(h_dst, k*k);

	dense_matrix<float, row_major, host_memory_space> dst2(d_dst.h(), d_dst.w());
	convert(dst2, d_dst);

	MAT_CMP(dst2, h_dst, 0.1);
}

BOOST_AUTO_TEST_CASE( copy_into_matrix )
{
	const int padding = 5;
	const int size = n + 2 * padding;

	dense_matrix<float, row_major, host_memory_space> h_pad(h_img.h(), size * size);
	dense_matrix<float, row_major, dev_memory_space> d_pad(d_img.h(), size * size);

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

	dense_matrix<int,row_major,host_memory_space> h_indices(c,o*o);
	dense_matrix<int,row_major,dev_memory_space> d_indices(c,o*o);

	max_pooling(h_pooled, h_img, p, 0, &h_indices);
	max_pooling(d_pooled, d_img, p, 0, &d_indices);

	dense_matrix<int, row_major, host_memory_space> indices2(d_indices.h(), d_indices.w());
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

BOOST_AUTO_TEST_CASE( max_pool_res )
{
	initialize_mersenne_twister_seeds();

	const int n = 64;
	int p = 3;
	int l = 2;
	const int m = (n-p)/(p-l)+1; // resulting image size
	const int c = 6;

	dense_matrix<float,row_major,host_memory_space> h_img(c,n*n);
	dense_matrix<float,row_major,host_memory_space> h_dst(c,m*m);
	dense_matrix<int,row_major,host_memory_space> h_indices(c,m*m);

	dense_matrix<float,row_major,dev_memory_space> d_img(c,n*n);
	dense_matrix<float,row_major,dev_memory_space> d_dst(c,m*m);
	dense_matrix<int,row_major,dev_memory_space> d_indices(c,m*m);

	fill_rnd_uniform(h_img.vec());
	convert(d_img, h_img);

	max_pooling(h_dst, h_img, p, l, &h_indices);
	max_pooling(d_dst, d_img, p, l, &d_indices);

	for(int k=0; k<c; k++) {
		for(int i=0; i<m; i++) {// loop through output image
			for(int j=0; j<m; j++) {
				float cmax = -FLT_MAX;
				for(int q=0; q<p; q++) { // loop through pool
					for(int r=0; r<p; r++) {
						int idx = (j*(p-l) + r) + (i*(p-l) + q)*n;
						if(cmax < h_img(k,idx))
							cmax = h_img(k,idx);
					}
				}
				BOOST_CHECK_EQUAL( h_dst(k,i*m+j), cmax );
			}
		}
	}

	MAT_CMP(d_dst, h_dst, 0.1);
	MAT_CMP(d_indices, h_indices, 0.1);

	// backprop step
	super_to_max(h_img, h_dst, p, l, &h_indices);
	super_to_max(d_img, d_dst, p, l, &d_indices);

	MAT_CMP(h_img, d_img, 0.1);
}


BOOST_AUTO_TEST_CASE( row_ncopy )
{
	sequence(d_img);
	sequence(h_img);

	d_img.resize(1, d_img.w()*d_img.h());
	h_img.resize(1, h_img.w()*h_img.h());

	int n=128;

	dense_matrix<float, row_major, host_memory_space> erg_h(n, h_img.w());
	dense_matrix<float, row_major, dev_memory_space> erg_d(n, d_img.w());
	fill(erg_d, 0.0f);
	fill(erg_h, 0.0f);
	for(int idx = 0; idx < erg_h.w(); idx++ ){
		for (int idy = 0; idy < n; idy++){
			erg_h.set(idy,idx, *(h_img.vec().ptr() + idx));
		}
	}


	cuv::row_ncopy(erg_d, d_img.vec(), n);

	for(int i=0;i<erg_h.h();i++){
		for(int j=0;j<erg_h.w();j++){
			BOOST_CHECK_CLOSE( erg_d(i,j), erg_h(i,j), 0.001 );
			if (i>1){
				BOOST_CHECK_CLOSE( erg_d(i,j), erg_d(i-1,j), 0.001 );
			}
		}
	}

}

BOOST_AUTO_TEST_CASE( strip_padding )
{

	sequence(d_img);
	//apply_scalar_functor(d_img,   SF_MULT,0.001f);

	int padding=2;

	int img_width 		= sqrt(d_img.w());
	int stripped_width  = img_width-2*padding;
	dense_matrix<float, row_major, host_memory_space> erg_h(d_img.h(), stripped_width*stripped_width);
	dense_matrix<float, row_major, dev_memory_space> erg_d(d_img.h(), stripped_width*stripped_width);
	fill(erg_d, 0.0f);
	fill(erg_h, 0.0f);

	cuv::strip_padding(erg_d, d_img, padding);

	int x,y, idx, idx_padded;
	float val;
	for (int img=0; img<d_img.h(); img++){
		for(int px=0; px<d_img.w(); px++){
			x = px % img_width;
			y = px / img_width;
			if ( x >=padding && x < padding+stripped_width &&
				 y >=padding && y < padding+stripped_width){
				idx 		=	y*img_width+x;
				idx_padded 	=	(y-padding)*stripped_width+(x-padding);

				val = d_img(img,idx);
				erg_h.set(img,idx_padded, val);
			}
		}
	}
	//std::cout << h_img ;

	for(int i=0;i<erg_h.h();i++){
		for(int j=0;j<erg_h.w();j++){
			BOOST_CHECK_CLOSE( erg_d(i,j), erg_h(i,j), 0.001 );
		}
	}
}

BOOST_AUTO_TEST_CASE( reverse_filters )
{




	dense_matrix<float, row_major, host_memory_space> filter_h(c*f, g*g);
	dense_matrix<float, row_major, dev_memory_space> filter_d(c*f, g*g);

	dense_matrix<float, row_major, host_memory_space> erg_h(c, f*g*g);
	dense_matrix<float, row_major, dev_memory_space> erg_d(c, f*g*g);

	fill(filter_h, 0.0f);
	fill(filter_d, 0.0f);
	fill(erg_h, 0.0f);
	fill(erg_d, 0.0f);

	vector<float,host_memory_space> one_filter_h(g*g);
	vector<float,dev_memory_space> one_filter_d(g*g);

	sequence(one_filter_h);
	sequence(one_filter_d);

	cuv::row_ncopy(filter_d, one_filter_d, c*f);
	cuv::row_ncopy(filter_h, one_filter_h,c*f);

	filter_d.resize(c, f*g*g);
	filter_h.resize(c, f*g*g);

	filter_inverse(erg_d,filter_d, g*g);

	float* f_h_ptr;
	int fs = g*g;
	int row_offset=0;
	f_h_ptr = filter_h.ptr();
	int f_h_w = filter_h.w();
	int numCases = filter_h.h();

	// iterate on every filter in a row
	for(int filter = 0; filter < f*g*g; filter = filter+g*g){
		// iterate on every element of the filter
		for(int y = 0; y < fs; y++){
			// every filterrow
			for(int nC = 0; nC <numCases; nC++){
				row_offset = nC*f_h_w;
				*(erg_h.ptr()+row_offset+filter+y) = *(f_h_ptr+row_offset+(fs-1)+filter-y);
			}

		}
	}

	std::cout << filter_d << std::endl << std::endl;



	std::cout << erg_d ;

	for(int i=0;i<erg_h.h();i++){
		for(int j=0;j<erg_h.w();j++){
			BOOST_CHECK_CLOSE( erg_d(i,j), erg_h(i,j), 0.001 );
		}
	}
}

BOOST_AUTO_TEST_CASE( add_maps )
{
	// c, n x n = 2, 64 x 64 - represents two (delta) maps that contribute to one destination delta map
	// the destination delta map is calculated as the sum of the individual delta maps
	sequence(h_img);
	sequence(d_img);

	// contains a map in each row where the summed pixels are stored
	dense_matrix<float, row_major, host_memory_space> erg_h(c, n);
	dense_matrix<float, row_major, dev_memory_space> erg_d(c, n);

	fill(erg_h, 0.0f);
	fill(erg_d, 0.0f);


	float* e_ptr = erg_h.ptr();
	float* i_ptr = h_img.ptr();
	int imagesize = 64;

	// host solution
	for (int row = 0; row<c; row++){
		for(int px = 0; px < n; px++){
			for(int img = 0; img < n; img++){
				*(e_ptr + row*erg_h.w() + px) += *(i_ptr + row*erg_h.w()    // mv to correct row in matrix
														 + img * imagesize  // iterate on image/delta maps
														 + px);				// iterate on pixels of destination map
			}
		}
	}

	add_maps_h(erg_d, d_img, n);

	std::cout << erg_d ;

	for(int i=0;i<erg_h.h();i++){
		for(int j=0;j<erg_h.w();j++){
			BOOST_CHECK_CLOSE( erg_d(i,j), erg_h(i,j), 0.001 );
		}
	}
}

}


