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





/*
 * bigimage_convolution_ops.cpp
 *
 *  Created on: 17.02.2010
 *      Author: gerharda
 */
#define BOOST_TEST_MODULE example
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <limits>

#include <cuv_general.hpp>
#include <dense_matrix.hpp>
#include <vector_ops.hpp>
#include <vector_ops/rprop.hpp>
#include <convolution_ops.hpp>
#include <convert.hpp>
#include <cv.h>
#include <highgui.h>

using namespace cuv;

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
	static const int c = 2;  // # patterns (images)
	static const int n = 1024;  // image size
	static const int f = 16;   // # filters
	static const int p = 8;	   // pooling size
	static const int g = 8;    // filter size
	static const int k = (n-g+1);//n-g+1;// target image size
	static const int o = n/p;  // pooling output size
	dense_matrix<float, row_major, dev_memory_space>  d_img,d_filter,d_dst,d_pooled;
	dense_matrix<float, row_major, host_memory_space> h_img,h_filter,h_dst,h_pooled;
	Fix()
		:   d_img(c,n*n), d_filter(c,f*g*g), d_dst(c,f*k*k), d_pooled(c,o*o)
		,   h_img(c,n*n), h_filter(c,f*g*g), h_dst(c,f*k*k), h_pooled(c,o*o)
	{
			//MEASURE_TIME("warmup", apply_scalar_functor(v, SF_EXP), 100);
	}
	~Fix(){
	}

};

BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( blur )
{
/*
 * Set up matrices - insert 1 pic 2 times in Matrix
 */
	std::vector<char*> fileName;
	fileName.push_back("/home/VI/stud/gerharda/Desktop/png-segments/001/frame000000_cam0.png");
	fileName.push_back("/home/VI/stud/gerharda/Desktop/png-segments/001/frame000001_cam0.png");

	IplImage* img[2];
	img[0] = cvLoadImage(fileName[0],0);
	if(!img) std::printf("Could not load image file: %s\n",fileName[0]);

	img[1] = cvLoadImage(fileName[1], 0);
	if(!img) std::printf("Could not load image file: %s\n",fileName[1]);

	int img_size = img[0]->width*img[0]->height*img[0]->nChannels;

	// TODO: make sure that h_img.w() == img->width()?
	float* h_vec = h_img.ptr();
	float* d_vec = d_img.ptr();

	char* le_ptr = img[0]->imageData;
	for (int i=0; i < img_size; i++ ){
		*h_vec++ = (float) (*le_ptr++);
	}
	le_ptr = img[1]->imageData;
	for (int i=0; i < img_size; i++ ){
		*h_vec++ = (float) (*le_ptr++);
	}
	convert(d_img, h_img);
/*
 * Set up filters
 */

	fill(d_filter.vec(), 0.0f);
	fill(h_filter.vec(), 0.0f);
	fill(d_dst.vec(), 0.0f);
	// in the middle of each filter i want to be a 1
	for (unsigned int i=g*g/2;
			i < c*g*g*f;
			i= i + g*g/2){
		*(h_filter.ptr() + i) = 1;
	}
	convert(d_filter,h_filter);

//	NVMatrix nv_dst(dst.ptr(), dst.h(), dst.w(), false);
//	NVMatrix nv_img(img.ptr(), img.h(), img.w(), false);
//	NVMatrix nv_filter(filter.ptr(), filter.h(), filter.w(), false);

//	std::cout << h_filter << std::endl;

    convolve2(d_dst,d_img,d_filter, f);
//	convolve(h_dst,h_img,h_filter);
//
	dense_matrix<float, row_major, host_memory_space> dst2(d_dst.h(), d_dst.w());
    convert(dst2,d_dst);

    //cvNamedWindow("org", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("transformed 0", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("transformed 1", CV_WINDOW_AUTOSIZE);

    //cvShowImage("org", img);

    img[0]->width = k;
    img[0]->height = k;
    img[0]->imageSize = k*k;
    img[0]->widthStep = k;

    img[1]->width = k;
    img[1]->height = k;
    img[1]->imageSize = k*k;
    img[1]->widthStep = k;

    // read back and show pic 0
    char* char_ptr = img[0]->imageData;
    memset(char_ptr, 0, img_size);
    float* dst2_ptr = dst2.ptr();
    for (int i=0; i < k*k; i++ ){
    	(*char_ptr++) = floor(((*dst2_ptr++)));
    }

    cvSaveImage("./filtered0.png", img[0]);
    cvShowImage("transformed 0", img[0]);
    cvWaitKey(0);
    // read back and show pic 1
    char_ptr = img[1]->imageData;
    memset(char_ptr, 0, img_size);

        //float* dst2_ptr = dst2.ptr();
    for (int i=0; i < k*k; i++ ){
        	(*char_ptr++) = floor(((*dst2_ptr++)));
        }
    //cvShowImage("transformed 1", img[1]);

//    cvSaveImage("./filtered0.png", img[0]);
//    cvSaveImage("./filtered1.png", img[1]);
    cvWaitKey(0);
    cvReleaseImage( &img[0] );
    cvReleaseImage( &img[1] );
    //cvDestroyWindow( "org" );
    cvDestroyWindow( "transformed 0" );
    cvDestroyWindow( "transformed 1" );

}

BOOST_AUTO_TEST_SUITE_END()

