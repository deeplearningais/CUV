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
#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>
#include <vector_ops.hpp>
#include <vector_ops/rprop.hpp>
#include <convolution_ops.hpp>
#include <convert.hpp>
#include <cv.h>
#include <highgui.h>

using namespace cuv;

struct Fix{
	static const int c = 1;  // # patterns (images)
	static const int n = 1024;  // image size
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


BOOST_AUTO_TEST_CASE( blur )
{
/*
 * Set up matrices - insert 1 pic 2 times in Matrix
 */
	char fileName[] = "/home/VI/stud/gerharda/lhome/workspace/SimpleCUDATestcase/build/hubble.png";
	IplImage* img = cvLoadImage(fileName,0);
	if(!img) std::printf("Could not load image file: %s\n",fileName);
	int img_size = img->width*img->height*img->nChannels;

	// TODO: what if h_img.w() != img->width()?
	float* h_vec = h_img.ptr();
	float* d_vec = d_img.ptr();
	char* le_ptr = img->imageData;
	for (int i=0; i < img_size; i++ ){
		*h_vec++ = (float) (*le_ptr++);
		//*d_vec++ = (float) (*le_ptr++);
	}
//	le_ptr = img->imageData;
//	for (int i=0; i < img_size; i++ ){
//		*h_vec++ = (float) (*le_ptr++);
//		//*d_vec++ = (float) (*le_ptr++);
//	}
	convert(d_img, h_img);

/*
 * Set up filters
 */

	fill(d_filter.vec(), 0.0f);
	fill(h_filter.vec(), 0.0f);

	// in the middle of each filter i want to be a 1
	for (unsigned int i=p*p/2;
			i < p*p*f;
			i= i + p*p/2){
		//*(d_filter.ptr() + i) = 1;
		*(h_filter.ptr() + i) = 1;
	}
	convert(d_filter,h_filter);



//	std::cout << h_filter << std::endl;

    convolve(d_dst,d_img,d_filter);
//	convolve(h_dst,h_img,h_filter);
//
	host_dense_matrix<float, row_major> dst2(d_dst.h(), d_dst.w());
    convert(dst2,d_dst);

    //cvNamedWindow("org", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("transformed", CV_WINDOW_AUTOSIZE);
    //cvShowImage("org", img);
    char* char_ptr = img->imageData;
    float* dst2_ptr = dst2.ptr();
    for (int i=0; i < img_size; i++ ){
    	(*char_ptr++) = (char) ((*dst2_ptr++));
    }
    cvShowImage("transformed", img);


    cvSaveImage("./filtered.png", img);
    cvWaitKey(0);
    cvReleaseImage( &img );
    //cvDestroyWindow( "org" );
    cvDestroyWindow( "transformed" );

}




BOOST_AUTO_TEST_SUITE_END()

