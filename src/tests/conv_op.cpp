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
#include <cmath>

#include <cuv/tools/cuv_test.hpp>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/tools/timing.hpp>
#include <cuv/basics/tensor.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/convolution_ops/convolution_ops.hpp>
#include <cuv/random/random.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/convolution_ops/convolution_ops_theano.hpp>
#define MEASURE_TIME(MSG, OPERATION, ITERS)     \
	float MSG;                                  \
	if(1){                                      \
		Timing tim;                             \
		for(int i=0;i<ITERS;i++){               \
			OPERATION ;                         \
		}                                       \
        safeThreadSync();                       \
		tim.update(ITERS);                      \
		printf("%s [%s] took %4.4f us/pass\n", #MSG, #OPERATION, 1000000.0f*tim.perf()); \
		MSG = 1000000.0f*tim.perf();            \
	}

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
	Fix()
	{
		//MEASURE_TIME("warmup", apply_scalar_functor(v, SF_EXP), 100);
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )

BOOST_AUTO_TEST_CASE( test_reorder_for_conv )
{
    using namespace cuv::alex_conv;
	unsigned int nImgChan = 7;      // must be divisible by nGroups
	unsigned int nImgPix  = 5;
	unsigned int nImg     = 2;

    tensor<float,dev_memory_space,row_major> inp(cuv::extents[nImg][nImgChan][nImgPix*nImgPix]);
	tensor<float,dev_memory_space,row_major> src(cuv::extents[nImgChan][nImgPix*nImgPix][nImg]);

    tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nImg][nImgChan][nImgPix*nImgPix]);
	tensor<float,host_memory_space,row_major> src_h(cuv::extents[nImgChan][nImgPix*nImgPix][nImg]);

    tensor<float,dev_memory_space,row_major> inp2(cuv::extents[nImg][nImgChan][nImgPix*nImgPix]);
    tensor<float,host_memory_space,row_major> inp2_h(cuv::extents[nImg][nImgChan][nImgPix*nImgPix]);

    sequence(inp);
    sequence(inp_h);
    src = 0.f;

    MEASURE_TIME(host_reorder,cuv::alex_conv::reorder_for_conv(src_h,inp_h), 2);
    MEASURE_TIME(dev_reorder, cuv::alex_conv::reorder_for_conv(src,inp), 2);
    cuvAssert(inp.shape(0)==src.shape(2));
    cuvAssert(inp.shape(1)==src.shape(0));
    cuvAssert(inp.shape(2)==src.shape(1));
    unsigned int cnt=0;
    for(unsigned int i=0;i<nImg;i++)
        for(unsigned int j=0;j<nImgChan;j++)
            for(unsigned int k=0;k<nImgPix*nImgPix;k++){
                BOOST_CHECK_EQUAL(inp(i,j,k), src(j,k,i));
            }
    MEASURE_TIME(host_reorder2,cuv::alex_conv::reorder_from_conv(inp2,src),2);
    MEASURE_TIME(dev_reorder2,cuv::alex_conv::reorder_from_conv(inp2_h,src_h),2);
    BOOST_CHECK_EQUAL(inp.ndim(),inp2.ndim());
    BOOST_CHECK_EQUAL(inp.shape(0),inp2.shape(0));
    BOOST_CHECK_EQUAL(inp.shape(1),inp2.shape(1));
    BOOST_CHECK_EQUAL(inp.shape(2),inp2.shape(2));
    BOOST_CHECK_EQUAL(inp_h.ndim(),inp2_h.ndim());
    BOOST_CHECK_EQUAL(inp_h.shape(0),inp2_h.shape(0));
    BOOST_CHECK_EQUAL(inp_h.shape(1),inp2_h.shape(1));
    BOOST_CHECK_EQUAL(inp_h.shape(2),inp2_h.shape(2));
    for(unsigned int i=0;i<nImg;i++)
        for(unsigned int j=0;j<nImgChan;j++)
            for(unsigned int k=0;k<nImgPix*nImgPix;k++){
                BOOST_CHECK_EQUAL(inp(i,j,k), inp2(i,j,k));
                BOOST_CHECK_EQUAL(inp_h(i,j,k), inp2_h(i,j,k));
            }
}

BOOST_AUTO_TEST_CASE( test_conv2d_hostdev )
{
    using namespace cuv::alex_conv;
	unsigned int nImgChan = 1;      // must be divisible by nGroups
	unsigned int nImgPix  = 16;
	unsigned int nImg     = 1;
    unsigned int nGroups  = 1;      // must be divisible by 2 ??
   
	unsigned int nFiltChan = nImgChan/nGroups;
	unsigned int nFiltPix  = 3;
	unsigned int nFilt     = 16; 

    unsigned int nResPix   = nImgPix+1-nFiltPix;


    tensor<float,dev_memory_space,row_major> inp(cuv::extents[nImg][nImgChan][nImgPix*nImgPix]);
	tensor<float,dev_memory_space,row_major> src(cuv::extents[nImgChan][nImgPix*nImgPix][nImg]);
	tensor<float,dev_memory_space,row_major> dst(cuv::extents[nFilt][nResPix*nResPix][nImg]);
	tensor<float,dev_memory_space,row_major> flt(cuv::extents[nFiltChan][nFiltPix*nFiltPix][nFilt]);
    cuv::alex_conv::reorder_for_conv(src,inp);

    for(unsigned int i=0;i<inp.size();i++) inp[i] = -0.1 + drand48();
    for(unsigned int i=0;i<flt.size();i++) flt[i] = -0.1 + drand48();
    dst = 0.f;

	tensor<float,host_memory_space,row_major> hsrc(cuv::extents[nImgChan][nImgPix*nImgPix][nImg]);
	tensor<float,host_memory_space,row_major> hdst(cuv::extents[nFilt][nResPix*nResPix][nImg]);
	tensor<float,host_memory_space,row_major> hflt(cuv::extents[nFiltChan][nFiltPix*nFiltPix][nFilt]);
    hsrc=src;
    hdst=dst;
    hflt=flt;

    MEASURE_TIME(conv_dev,         convolve2d(dst,src,flt, 0, 1, nGroups), 10);
    MEASURE_TIME(conv_hst,         convolve2d(hdst,hsrc,hflt, 0, 1, nGroups), 10);

    for(unsigned int i=0;i<hdst.shape(0);i++)
        for (unsigned int j = 0; j < hdst.shape(1); ++j)
            for (unsigned int k = 0; k < hdst.shape(2); ++k)
            {
                BOOST_CHECK_CLOSE((float)dst(i,j,k),(float)hdst(i,j,k),0.1f);
            }

    // check derivative w.r.t. images
    for(unsigned int i=0;i<hdst.size();i++) hdst[i] = -0.1 + drand48();
    hdst = 0.f; hdst[0]=1.f;
    dst = hdst; 
    flt=2.f; hflt = flt;
    src = 0.f;
    hsrc = 0.f;
    MEASURE_TIME(d_conv_dimg_dev,           d_conv2d_dimg(src,dst,flt, 0, 1, nGroups), 10);
    MEASURE_TIME(d_conv_dimg_hst,  hsrc=0.f;d_conv2d_dimg(hsrc,hdst,hflt, 0, 1, nGroups), 10);

    for(unsigned int i=0;i<src.shape(1);i++)
        std::cout <<src[i]<<" ";
    std::cout << "norm2 of gradient: "<<cuv::norm2(src)<<std::endl;

    for(unsigned int i=0;i<hsrc.shape(0);i++)
        for (unsigned int j = 0; j < hsrc.shape(1); ++j)
            for (unsigned int k = 0; k < hsrc.shape(2); ++k)
            {
                BOOST_CHECK_CLOSE((float)src(i,j,k),(float)hsrc(i,j,k),0.01f);
            }
}


BOOST_AUTO_TEST_CASE( test_conv2d )
{
    using namespace cuv::alex_conv;
	unsigned int nImgChan = 8;      // must be divisible by nGroups
	unsigned int nImgPix  = 176;
	unsigned int nImg     = 16;
    unsigned int nGroups  = 1;      // must be divisible by 2 ??
   
   unsigned int nFiltChan = nImgChan/nGroups;
   unsigned int nFiltPix  = 7;
   unsigned int nFilt     = 32; 

   unsigned int nResPix   = nImgPix-nFiltPix+1;

   tensor<float,dev_memory_space,row_major> inp(cuv::extents[nImg][nImgChan][nImgPix][nImgPix]);

   tensor<float,dev_memory_space,row_major> src(cuv::extents[nImgChan][nImgPix][nImgPix][nImg]);
   tensor<float,dev_memory_space,row_major> dst(cuv::extents[nFilt][nResPix][nResPix][nImg]);

   tensor<float,dev_memory_space,row_major> flt(cuv::extents[nFiltChan][nFiltPix*nFiltPix][nFilt]);

   cuv::alex_conv::reorder_for_conv(src,inp);

   //convolve2d(tensor<float,dev_memory_space>& dst, 
   //        const tensor<float,dev_memory_space>& img, 
   //        const tensor<float,dev_memory_space>& filter,
   //        unsigned int nModulesX,
   //        unsigned int paddingStart, 
   //        unsigned int moduleStride,
   //        unsigned int nGroups){

   MEASURE_TIME(conv_dev,         convolve2d(dst,src,flt, 0, 1, nGroups), 10);
   MEASURE_TIME(d_conv_dimg_dev,  d_conv2d_dimg(src,dst,flt, 0, 1, nGroups), 10);
   MEASURE_TIME(d_conv_dfilt_dev, d_conv2d_dfilt(flt,dst,src, 0, 1, nGroups,4), 10);

}

BOOST_AUTO_TEST_CASE( test_conv2d_sparse )
{

    // the idea here is that if we set the connections right, the sparse
    // operation should do exactly the same as the dense operation.

    using namespace cuv::alex_conv;
	unsigned int nImgChan = 32;      // must be divisible by nGroups
	unsigned int nImgPix  = 16;
	unsigned int nImg     = 16;
    unsigned int nGroups  = 2;      // must be divisible by 2 ??
   
   unsigned int nFiltChan = nImgChan/nGroups;
   unsigned int nFiltPix  = 7;
   unsigned int nFilt     = 32; 

   unsigned int nResPix   = nImgPix-nFiltPix+1;

   // in sparse filters, the destination layer is grouped into equal-sized blocks
   // of consecutive filters.
   //
   // each of these blocks is then connected to a random subset of nodes in the
   // source layer.
   //
   // here, we choose two groups arbitrarily and require that each group is
   // connects to all input maps, in order. The result should be equivalent to
   // a "dense" connection of maps.
   unsigned int nSparseGroups = nGroups;
   unsigned int oversample = nGroups * nFiltChan / nImgChan;
   tensor<int, dev_memory_space, row_major> indices(cuv::extents[nSparseGroups][oversample * nImgChan]);
   for(unsigned int i=0; i < nSparseGroups; i++){
       std::vector<unsigned int> v(nImgChan);
       for (unsigned int k = 0; k < v.size(); ++k)
           v[k] = k;
       // we shouldn't shuffle in this test to get same result as dens connection
       //std::random_shuffle(v.begin(), v.end());
       for (unsigned int o = 0; o < oversample; ++o)
       {
           for (unsigned int k = 0; k < nImgChan; ++k)
           {
               indices(i, o*nImgChan + k) = v[k];
           }
       }
   }

   tensor<float,dev_memory_space,row_major> src(cuv::extents[nImgChan][nImgPix][nImgPix][nImg]);
   tensor<float,dev_memory_space,row_major> src_sparse(cuv::extents[nImgChan][nImgPix][nImgPix][nImg]);

   tensor<float,dev_memory_space,row_major> dst(cuv::extents[nFilt][nResPix][nResPix][nImg]);
   tensor<float,dev_memory_space,row_major> dst_sparse(cuv::extents[nFilt][nResPix][nResPix][nImg]);

   tensor<float,dev_memory_space,row_major> flt(cuv::extents[nFiltChan][nFiltPix*nFiltPix][nFilt]);
   tensor<float,dev_memory_space,row_major> flt_sparse(cuv::extents[nFiltChan][nFiltPix*nFiltPix][nFilt]);

   sequence(src);
   sequence(flt);
   //fill_rnd_uniform(src);
   //fill_rnd_uniform(flt);

   //convolve2d(tensor<float,dev_memory_space>& dst, 
   //        const tensor<float,dev_memory_space>& img, 
   //        const tensor<float,dev_memory_space>& filter,
   //        unsigned int nModulesX,
   //        unsigned int paddingStart, 
   //        unsigned int moduleStride,
   //        unsigned int nGroups){

   if(1){
       convolve2d(dst_sparse,src,flt,indices, 0, 1, nSparseGroups);
       safeThreadSync();
       convolve2d(dst,src,flt, 0, 1, nGroups);
       safeThreadSync();
       BOOST_CHECK_LT(norm2(dst-dst_sparse), 0.001f);
   }

    //d_conv2d_dimg( dst,  delta,  filter,
    //        int paddingStart=0, unsigned int moduleStride=0, unsigned int nGroups=0, float factNew=1.f, float factOld=0.f);
   if(1){
       d_conv2d_dimg(src,dst,flt, 0, 1, nGroups);
       safeThreadSync();
       d_conv2d_dimg(src_sparse, dst, flt, indices, 0, 1, nSparseGroups);
       safeThreadSync();
       BOOST_CHECK_LT(norm2(src-src_sparse), 0.001f);
   }

   if(1){
       d_conv2d_dfilt(flt,dst,src, 0, 1, nGroups, 4);
       safeThreadSync();
       flt_sparse = 0.f;
       d_conv2d_dfilt(flt_sparse,dst,src, indices, 0, 1, nSparseGroups, 4);
       safeThreadSync();
       BOOST_CHECK_LT(norm2(flt-flt_sparse), 0.001f);
   }
}

BOOST_AUTO_TEST_CASE( test_conv2d_fail )
{
    // regression test for a cuvnet problem that appears to be
    // non-determininstic -- it only occurs if other processes are running on
    // the same device.
   
    using namespace cuv::alex_conv;
	unsigned int nImgChan = 3;      // must be divisible by nGroups
	unsigned int nImgPix  = 16;
	unsigned int nImg     = 4;
    unsigned int nGroups  = 1;      // must be divisible by 2 ??
   
   unsigned int nFiltChan = nImgChan/nGroups;
   unsigned int nFiltPix  = 3;
   unsigned int nFilt     = 16; 

   unsigned int nResPix   = nImgPix-nFiltPix+1;

   tensor<float,dev_memory_space,row_major> inp(cuv::extents[nImg][nImgChan][nImgPix][nImgPix]);

   tensor<float,dev_memory_space,row_major> src(cuv::extents[nImgChan][nImgPix][nImgPix][nImg]);
   tensor<float,dev_memory_space,row_major> dst(cuv::extents[nFilt][nResPix][nResPix][nImg]);

   tensor<float,dev_memory_space,row_major> flt(cuv::extents[nFiltChan][nFiltPix*nFiltPix][nFilt]);


   //convolve2d(tensor<float,dev_memory_space>& dst, 
   //        const tensor<float,dev_memory_space>& img, 
   //        const tensor<float,dev_memory_space>& filter,
   //        unsigned int nModulesX,
   //        unsigned int paddingStart, 
   //        unsigned int moduleStride,
   //        unsigned int nGroups){

   //for(unsigned int i=0; i<100; i++)
       //fill_rnd_uniform(inp);
   for(int i=0; i<1000; i++){
       fill_rnd_uniform(inp);
       fill_rnd_uniform(flt);
       std::cout << "cuv::sum(inp):" << cuv::sum(inp) << std::endl;
       cuv::alex_conv::reorder_for_conv(src,inp);
       cuvAssert(!cuv::has_nan(src));
       cuvAssert(!cuv::has_inf(src));
       cuvAssert(!cuv::has_nan(flt));
       cuvAssert(!cuv::has_inf(flt));
       dst = NAN;
       convolve2d(dst,src,flt, 0, 1, nGroups);
       cuvAssert(!cuv::has_nan(dst));
       cuvAssert(!cuv::has_inf(dst));
       std::cout << "cuv::sum(dst):" << cuv::sum(dst) << std::endl;
       src = NAN;
       d_conv2d_dimg(src,dst,flt, 0, 1, nGroups);
       std::cout << "cuv::sum(src):" << cuv::sum(src) << std::endl;
       cuvAssert(!cuv::has_nan(src));
       cuvAssert(!cuv::has_inf(src));
       flt = NAN;
       fill_rnd_uniform(dst);
       fill_rnd_uniform(src);
       d_conv2d_dfilt(flt,dst,src, 0, 1, nGroups, 0);
       std::cout << "cuv::sum(flt):" << cuv::sum(flt) << std::endl;
       cuvAssert(!cuv::has_nan(flt));
       cuvAssert(!cuv::has_inf(flt));
   }

}

BOOST_AUTO_TEST_CASE( test_conv2d_theano )
{
  using namespace cuv::theano_conv;
  unsigned int nImg  =  32;
  unsigned int nMaps =  8;
  unsigned int imgH  = 176, imgW=176;
  unsigned int nFilt = 32;
  unsigned int fsX   = 7, fsY = 7;
  std::string mode   = "valid";
  unsigned int dstH = mode == "full" ? imgH+fsY-1 : imgH-fsY+1;
  unsigned int dstW = mode == "full" ? imgW+fsX-1 : imgW-fsX+1;
  // images: (nImg,nMaps,imgH,imgW)
  // out   : (nImg,nFilt,imgH-fsY+1,imgW-fsX+1)
  // kern  : (nFilt,nMaps,fsY,fsX)
  cuv::tensor<float,cuv::dev_memory_space> images(cuv::extents[nImg][nMaps][imgH][imgW]);
  cuv::tensor<float,cuv::dev_memory_space> kern(cuv::extents[nFilt][nMaps][fsY][fsX]);
  cuv::tensor<float,cuv::dev_memory_space> out(cuv::extents[nImg][nFilt][dstH][dstW]);
  images = 0.001f;
  kern = 0.001f;
  out = 0.001f;

  initcuda();
  timeval a, b;

  std::cout << "--------------------------------------- CONVOLVE ---------------"<<std::endl;
for(int ver=-1;ver<0;ver++){
      gettimeofday(&a, 0);
      for(unsigned int i=0;i<10;i++){
          convolve_2d(out,images,kern, mode,ver);
      }
      gettimeofday(&b, 0);
      //printdiff(a,b,10);
}

  std::cout << "--------------------------------------- DIMG ---------------"<<std::endl;
  gettimeofday(&a, 0);
  for(unsigned int i=0;i<10;i++){
      d_convolve_d_images(images,out,kern, mode);
  }
  gettimeofday(&b, 0);
  //printdiff(a,b,10);

  std::cout << "--------------------------------------- DKRN ---------------"<<std::endl;
  gettimeofday(&a, 0);
  for(unsigned int i=0;i<10;i++){
      d_convolve_d_kern(kern,images, out, mode);
  }
  gettimeofday(&b, 0);
  //printdiff(a,b,10);

  /*PyArrayObject* pa = (PyArrayObject*)CudaNdarray_CreateArrayObj(cnda);*/
  /*print_numeric_array(pa);*/

  finalize_cuda();
}







BOOST_AUTO_TEST_CASE( test_conv1d_theano )
{
  using namespace cuv::theano_conv;

  {
      unsigned int nImg  =  2;
      unsigned int nMaps =  20;
      unsigned int imgH  = 1, imgW=5;
      unsigned int nFilt = 2;
      unsigned int fsX  = 2, fsY = 1;
      unsigned int dstH = imgH-fsY+1;
      unsigned int dstW = imgW-fsX+1;
      // images: (nImg,nMaps,imgH,imgW)
      // out   : (nImg,nFilt,imgH-fsY+1,imgW-fsX+1)
      // kern  : (nFilt,nMaps,fsY,fsX)
      cuv::tensor<float,cuv::dev_memory_space> images(cuv::extents[nImg][nMaps][imgH][imgW]);
      cuv::tensor<float,cuv::dev_memory_space> kern(cuv::extents[nFilt][nMaps][fsY][fsX]);
      cuv::tensor<float,cuv::dev_memory_space> out(cuv::extents[nImg][nFilt][dstH][dstW]);

      initcuda();

      std::string mode = "valid";
      timeval a, b;

      for(int ver=-1;ver<0;ver++){
          gettimeofday(&a, 0);
          for(unsigned int i=0;i<10;i++){
              convolve_2d(out,images,kern, mode,ver);
          }
          gettimeofday(&b, 0);
          //printdiff(a,b,10);
      }

      std::cout << "--------------------------------------- DIMG ---------------"<<std::endl;
      gettimeofday(&a, 0);
      for(unsigned int i=0;i<10;i++){
          d_convolve_d_images(images,out,kern, mode);
      }
      gettimeofday(&b, 0);
      //printdiff(a,b,10);

      std::cout << "--------------------------------------- DKRN ---------------"<<std::endl;
      gettimeofday(&a, 0);
      for(unsigned int i=0;i<10;i++){
          d_convolve_d_kern(kern,images, out, mode);
      }
      gettimeofday(&b, 0);
      //printdiff(a,b,10);

  }

  finalize_cuda();
}


BOOST_AUTO_TEST_CASE( test_tuplewise_op )
{
     std::cout << "testing tuplewise op started" << std::endl;/* cursor */
   using namespace cuv::alex_conv;

   // norm test
   {
     unsigned int sub_size = 4;
     unsigned int nImg = 4;
     unsigned int nPix = 16;
     unsigned int nChan = 8*sub_size;
     tensor<float,dev_memory_space,row_major> inp(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> res(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> res_h(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     fill_rnd_uniform(inp_h);
     inp = inp_h;

     fill_rnd_uniform(res_h);
     res = res_h;
     tuplewise_op(res,inp, 0, sub_size);
     tuplewise_op(res_h,inp_h, 0, sub_size);

     for(unsigned int i=0;i<nChan/sub_size;i++)
         for(unsigned int j=0;j<nPix;j++)
             for(unsigned int k=0;k<nPix;k++){
                 for(unsigned int l=0;l<nImg;l++){
                     BOOST_CHECK_CLOSE(1.f, 1.f + res(i,j,k,l) - res_h(i,j,k,l), 0.001f);

                     float squared_sum = 0.f;
                     for(unsigned int sub_idx = 0; sub_idx < sub_size; sub_idx++){
                         float s = inp_h(sub_size*i+sub_idx,j,k,l);
                         squared_sum += s*s;
                     }
                     BOOST_CHECK_CLOSE(1.f, 1.f + res_h(i,j,k,l)-sqrt(squared_sum), 0.001f);
                 }
             }
   }
   {
   unsigned int sub_size = 5;
   unsigned int nImg = 4;
   unsigned int nPix = 16;
    unsigned int nChan = 8*sub_size;
   tensor<float,dev_memory_space,row_major> inp(cuv::extents[nPix][nPix][nImg][nChan]);
   tensor<float,dev_memory_space,row_major> res(cuv::extents[nPix][nPix][nImg][nChan/sub_size]);

   tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nPix][nPix][nImg][nChan]);
   tensor<float,host_memory_space,row_major> res_h(cuv::extents[nPix][nPix][nImg][nChan/sub_size]);

   fill_rnd_uniform(inp_h);
   inp = inp_h;

   fill_rnd_uniform(res_h);
   res = res_h;
   tuplewise_op(res,inp,3, sub_size);
   tuplewise_op(res_h,inp_h,3,sub_size);

   for(unsigned int i=0;i<nPix;i++)
      for(unsigned int j=0;j<nPix;j++)
          for(unsigned int k=0;k<nImg;k++){
              for(unsigned int l=0;l<nChan/sub_size;l++){
                  BOOST_CHECK_CLOSE(1.f, 1.f + res(i,j,k,l) - res_h(i,j,k,l), 0.001f);

                  float squared_sum = 0.f;
                  for(unsigned int sub_idx = 0; sub_idx < sub_size; sub_idx++){
                      float s = inp_h(i,j,k,sub_size*l+sub_idx);
                      squared_sum += s*s;
                  }
                  BOOST_CHECK_CLOSE(1.f, 1.f + res_h(i,j,k,l)-sqrt(squared_sum), 0.001f);
              }
          }

   }
   {
       unsigned int sub_size = 7;
       unsigned int nImg = 30;
   // norm test
   {
     unsigned int sub_size = 4;
     unsigned int nImg = 4;
     unsigned int nPix = 16;
     unsigned int nChan = 8*sub_size;
     tensor<float,dev_memory_space,row_major> inp(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> res(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> res_h(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     fill_rnd_uniform(inp_h);
     inp = inp_h;

     fill_rnd_uniform(res_h);
     res = res_h;
     tuplewise_op(res,inp, 0, sub_size);
     tuplewise_op(res_h,inp_h, 0, sub_size);

     for(unsigned int i=0;i<nChan/sub_size;i++)
         for(unsigned int j=0;j<nPix;j++)
             for(unsigned int k=0;k<nPix;k++){
                 for(unsigned int l=0;l<nImg;l++){
                     BOOST_CHECK_CLOSE(1.f, 1.f + res(i,j,k,l) - res_h(i,j,k,l), 0.001f);

                     float squared_sum = 0.f;
                     for(unsigned int sub_idx = 0; sub_idx < sub_size; sub_idx++){
                         float s = inp_h(sub_size*i+sub_idx,j,k,l);
                         squared_sum += s*s;
                     }
                     BOOST_CHECK_CLOSE(1.f, 1.f + res_h(i,j,k,l)-sqrt(squared_sum), 0.001f);
                 }
             }
   }
       unsigned int nChan = 8*sub_size;
       tensor<float,dev_memory_space,row_major> inp(cuv::extents[nImg][nChan]);
       tensor<float,dev_memory_space,row_major> res(cuv::extents[nImg][nChan/sub_size]);

       tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nImg][nChan]);
       tensor<float,host_memory_space,row_major> res_h(cuv::extents[nImg][nChan/sub_size]);

       fill_rnd_uniform(inp_h);
       inp = inp_h;

       fill_rnd_uniform(res_h);
       res = res_h;
       tuplewise_op(res,inp,1, sub_size);
       tuplewise_op(res_h,inp_h,1, sub_size);

       for(unsigned int k=0;k<nImg;k++){
           for(unsigned int l=0;l<nChan/sub_size;l++){
               BOOST_CHECK_CLOSE(1.f, 1.f + res(k,l) - res_h(k,l), 0.001f);

               float squared_sum = 0.f;
               for(unsigned int sub_idx = 0; sub_idx < sub_size; sub_idx++){
                   int index = sub_size*l+sub_idx;
                   float s = inp_h(k,index);
                   squared_sum += s*s;
               }
               BOOST_CHECK_CLOSE(1.f, 1.f + res_h(k,l)-sqrt(squared_sum), 0.001f);
           }
       }

   }
   // max out test dim = 0
   {
     unsigned int sub_size = 4;
     unsigned int nImg = 4;
     unsigned int nPix = 16;
     unsigned int nChan = 8*sub_size;
     tensor<float,dev_memory_space,row_major> inp(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> res(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> res_h(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     fill_rnd_uniform(inp_h);
     inp = inp_h;

     fill_rnd_uniform(res_h);
     res = res_h;
     tuplewise_op(res,inp, 0, sub_size, TO_MAX);
     tuplewise_op(res_h,inp_h, 0, sub_size, TO_MAX);

     for(unsigned int i=0;i<nChan/sub_size;i++)
         for(unsigned int j=0;j<nPix;j++)
             for(unsigned int k=0;k<nPix;k++){
                 for(unsigned int l=0;l<nImg;l++){
                     BOOST_CHECK_CLOSE(1.f, 1.f + res(i,j,k,l) - res_h(i,j,k,l), 0.001f);

                     float max_act = inp_h(sub_size*i,j,k,l);
                     for(unsigned int sub_idx = 1; sub_idx < sub_size; sub_idx++){
                        float s = inp_h(sub_size*i+sub_idx,j,k,l);
                        if(max_act < s){
                            max_act = s;
                        }
                     }
                     BOOST_CHECK_CLOSE(1.f, 1.f + res_h(i,j,k,l)- max_act, 0.001f);
                 }
             }
   }
   // max out test dim = last
   {
       unsigned int sub_size = 4;
       unsigned int nImg = 4;
       unsigned int nPix = 16;
       unsigned int nChan = 8*sub_size;
       tensor<float,dev_memory_space,row_major> inp(cuv::extents[nImg][nPix][nPix][nChan]);
       tensor<float,dev_memory_space,row_major> res(cuv::extents[nImg][nPix][nPix][nChan/sub_size]);

       tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nImg][nPix][nPix][nChan]);
       tensor<float,host_memory_space,row_major> res_h(cuv::extents[nImg][nPix][nPix][nChan/sub_size]);

       fill_rnd_uniform(inp_h);
       inp = inp_h;

       fill_rnd_uniform(res_h);
       res = res_h;
       tuplewise_op(res,inp, 3, sub_size, TO_MAX);
       tuplewise_op(res_h,inp_h, 3, sub_size, TO_MAX);

       for(unsigned int l=0;l<nImg;l++){
           for(unsigned int j=0;j<nPix;j++)
               for(unsigned int k=0;k<nPix;k++){
                   for(unsigned int i=0;i<nChan/sub_size;i++){
                       BOOST_CHECK_CLOSE(1.f, 1.f + res(l,j,k,i) - res_h(l,j,k,i), 0.001f);

                       float max_act = inp_h(l,j,k,sub_size*i);
                       for(unsigned int sub_idx = 1; sub_idx < sub_size; sub_idx++){
                           float s = inp_h(l,j,k,sub_size*i+sub_idx);
                           if(max_act < s){
                               max_act = s;
                           }
                       }
                       BOOST_CHECK_CLOSE(1.f, 1.f + res_h(l,j,k,i)- max_act, 0.001f);
                   }
               }
       }
       std::cout << " test for max out dim last passed" << std::endl;/* cursor */
   }
   // max out test dim = last, 2-dim tensor
   {
       unsigned int sub_size = 4;
       unsigned int nImg = 4;
       unsigned int nChan = 8*sub_size;
       tensor<float,dev_memory_space,row_major> inp(cuv::extents[nImg][nChan]);
       tensor<float,dev_memory_space,row_major> res(cuv::extents[nImg][nChan/sub_size]);

       tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nImg][nChan]);
       tensor<float,host_memory_space,row_major> res_h(cuv::extents[nImg][nChan/sub_size]);

       fill_rnd_uniform(inp_h);
       inp = inp_h;

       fill_rnd_uniform(res_h);
       res = res_h;

       tuplewise_op(res,inp, 1, sub_size, TO_MAX);
       tuplewise_op(res_h,inp_h, 1, sub_size, TO_MAX);

       for(unsigned int l=0;l<nImg;l++){
           for(unsigned int i=0;i<nChan/sub_size;i++){
               BOOST_CHECK_CLOSE(1.f, 1.f + res(l,i) - res_h(l,i), 0.001f);

               float max_act = inp_h(l,sub_size*i);
               for(unsigned int sub_idx = 1; sub_idx < sub_size; sub_idx++){
                   float s = inp_h(l,sub_size*i+sub_idx);
                   if(max_act < s){
                       max_act = s;
                   }
               }
               BOOST_CHECK_CLOSE(1.f, 1.f + res_h(l,i)- max_act, 0.001f);
           }
       }
       std::cout << " test for max out dim last, 2-dim tensor passed" << std::endl;/* cursor */
   }

   // squared norm test
   {
     unsigned int sub_size = 4;
     unsigned int nImg = 4;
     unsigned int nPix = 16;
     unsigned int nChan = 8*sub_size;
     tensor<float,dev_memory_space,row_major> inp(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> res(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> res_h(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     fill_rnd_uniform(inp_h);
     inp = inp_h;

     fill_rnd_uniform(res_h);
     res = res_h;
     tuplewise_op(res,inp, 0, sub_size, TO_ADD_SQUARED);
     tuplewise_op(res_h,inp_h, 0, sub_size, TO_ADD_SQUARED);

     for(unsigned int i=0;i<nChan/sub_size;i++)
         for(unsigned int j=0;j<nPix;j++)
             for(unsigned int k=0;k<nPix;k++){
                 for(unsigned int l=0;l<nImg;l++){
                     BOOST_CHECK_CLOSE(1.f, 1.f + res(i,j,k,l) - res_h(i,j,k,l), 0.001f);

                     float squared_sum = 0.f;
                     for(unsigned int sub_idx = 0; sub_idx < sub_size; sub_idx++){
                         float s = inp_h(sub_size*i+sub_idx,j,k,l);
                         squared_sum += s*s;
                     }
                     BOOST_CHECK_CLOSE(1.f, 1.f + res_h(i,j,k,l)-squared_sum, 0.001f);
                 }
             }
   std::cout << "testing squared norm done" << std::endl;/* cursor */
   }
   // subsample test
   {
     std::cout << "testing subsampling started" << std::endl;/* cursor */
     unsigned int sub_size = 4;
     unsigned int nImg = 4;
     unsigned int nPix = 16;
     unsigned int nChan = 8*sub_size;
     tensor<float,dev_memory_space,row_major> inp(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> res(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> res_h(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     fill_rnd_uniform(inp_h);
     inp = inp_h;

     fill_rnd_uniform(res_h);
     res = res_h;
     tuplewise_op(res,inp, 0, sub_size, TO_SUBSAMPLE);
     tuplewise_op(res_h,inp_h, 0, sub_size, TO_SUBSAMPLE);

     for(unsigned int i=0;i<nChan/sub_size;i++)
         for(unsigned int j=0;j<nPix;j++)
             for(unsigned int k=0;k<nPix;k++){
                 for(unsigned int l=0;l<nImg;l++){
                     BOOST_CHECK_CLOSE(1.f, 1.f + res(i,j,k,l) - res_h(i,j,k,l), 0.001f);
                     BOOST_CHECK_CLOSE(1.f, 1.f + res_h(i,j,k,l)-inp_h(sub_size*i,j,k,l), 0.001f);
                 }
             }
     std::cout << "testing subsampling done" << std::endl;/* cursor */
   }
   // mean test
   {
     unsigned int sub_size = 4;
     unsigned int nImg = 4;
     unsigned int nPix = 16;
     unsigned int nChan = 8*sub_size;
     tensor<float,dev_memory_space,row_major> inp(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> res(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> res_h(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     fill_rnd_uniform(inp_h);
     inp = inp_h;

     fill_rnd_uniform(res_h);
     res = res_h;
     tuplewise_op(res,inp, 0, sub_size, TO_MEAN);
     tuplewise_op(res_h,inp_h, 0, sub_size, TO_MEAN);

     for(unsigned int i=0;i<nChan/sub_size;i++)
         for(unsigned int j=0;j<nPix;j++)
             for(unsigned int k=0;k<nPix;k++){
                 for(unsigned int l=0;l<nImg;l++){
                     BOOST_CHECK_CLOSE(1.f, 1.f + res(i,j,k,l) - res_h(i,j,k,l), 0.001f);

                     float squared_sum = 0.f;
                     for(unsigned int sub_idx = 0; sub_idx < sub_size; sub_idx++){
                         float s = inp_h(sub_size*i+sub_idx,j,k,l);
                         squared_sum += s;
                     }

                     BOOST_CHECK_CLOSE(1.f, 1.f + res_h(i,j,k,l)- squared_sum/sub_size, 0.001f);
                 }
             }
     std::cout << "testing mean done" << std::endl;/* cursor */
   }
}

BOOST_AUTO_TEST_CASE( test_tuplewise_op_grad )
{
   float eps = 0.0001f;
   {
       std::cout << "in gradient norm" << std::endl;
     using namespace cuv::alex_conv;
     unsigned int sub_size = 2;
     unsigned int nImg = 4;
     unsigned int nPix = 10;
     unsigned int nChan = 2 * sub_size;
     tensor<float,dev_memory_space,row_major> inp_grad(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> inp(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> res(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> delta(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     tensor<float,host_memory_space,row_major> inp_grad_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> res_h(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> delta_h(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     fill_rnd_uniform(inp_h);
     inp = inp_h;

     fill_rnd_uniform(delta_h);
     fill_rnd_uniform(res_h); // should be overwritten
     res = res_h; // copy to device
     fill_rnd_uniform(inp_grad_h); // should be overwritten
     inp_grad = inp_grad_h; // copy to device
     delta = delta_h;
     tuplewise_op(res,inp, 0, sub_size, TO_NORM, eps);
     tuplewise_op(res_h,inp_h, 0, sub_size,TO_NORM,eps);

     tuplewise_op_grad(inp_grad,inp,delta, 0, sub_size,TO_NORM, eps);
     tuplewise_op_grad(inp_grad_h,inp_h,delta_h, 0, sub_size,TO_NORM, eps);

     for(unsigned int i=0;i<nChan/sub_size;i++)
         for(unsigned int j=0;j<nPix;j++)
             for(unsigned int k=0;k<nPix;k++){
                 for(unsigned int l=0;l<nImg;l++){
                     for (unsigned int sub_idx = 0; sub_idx < sub_size; sub_idx++){
                         BOOST_CHECK_CLOSE(1.f, 1.f + inp_grad(sub_size*i+sub_idx,j,k,l) - inp_grad_h(sub_size*i+sub_idx,j,k,l), 0.001);

                         float s0 = inp_h(sub_size*i+sub_idx,j,k,l);
                         float r  = res_h(  i  ,j,k,l);
                         float d  = delta(  i  ,j,k,l);
                         float f0 = d / (r) * s0;
                         BOOST_CHECK_CLOSE(1.f, 1.f + inp_grad_h(sub_size*i + sub_idx,j,k,l) - f0, 0.01f);
                     }

                 }
             }
       std::cout << "in gradient norm finished"<< std::endl;
   }

   {
       std::cout << "in gradient norm (2)"<< std::endl;
    using namespace cuv::alex_conv;
    unsigned int sub_size = 3;
    unsigned int nImg = 4;
    unsigned int nPix = 16;
    unsigned int nChan = 8 * sub_size;
    tensor<float,dev_memory_space,row_major> inp_grad(cuv::extents[nPix][nPix][nImg][nChan]);
    tensor<float,dev_memory_space,row_major> inp(cuv::extents[nPix][nPix][nImg][nChan]);
    tensor<float,dev_memory_space,row_major> res(cuv::extents[nPix][nPix][nImg][nChan/sub_size]);
    tensor<float,dev_memory_space,row_major> delta(cuv::extents[nPix][nPix][nImg][nChan/sub_size]);

    tensor<float,host_memory_space,row_major> inp_grad_h(cuv::extents[nPix][nPix][nImg][nChan]);
    tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nPix][nPix][nImg][nChan]);
    tensor<float,host_memory_space,row_major> res_h(cuv::extents[nPix][nPix][nImg][nChan/sub_size]);
    tensor<float,host_memory_space,row_major> delta_h(cuv::extents[nPix][nPix][nImg][nChan/sub_size]);

    fill_rnd_uniform(inp_h);
    inp = inp_h;

    fill_rnd_uniform(res_h); // should be overwritten
    res = res_h; // copy to device
    fill_rnd_uniform(inp_grad_h); // should be overwritten
    inp_grad = inp_grad_h; // copy to device
    fill_rnd_uniform(delta_h);
    delta = delta_h;
    tuplewise_op(res,inp, 3, sub_size,TO_NORM,eps);
    tuplewise_op(res_h,inp_h, 3, sub_size,TO_NORM,eps);

    tuplewise_op_grad(inp_grad,inp,delta, 3, sub_size,TO_NORM, eps);
    tuplewise_op_grad(inp_grad_h,inp_h,delta_h, 3, sub_size,TO_NORM, eps);

    for(unsigned int i=0;i<nPix;i++)
        for(unsigned int j=0;j<nPix;j++)
            for(unsigned int k=0;k<nImg;k++){
                for(unsigned int l=0;l<nChan/sub_size;l++){
                    for (unsigned int sub_idx = 0; sub_idx < sub_size; sub_idx++){
                        BOOST_CHECK_CLOSE(1.f, 1.f + inp_grad(i,j,k,sub_size*l+sub_idx) - inp_grad_h(i,j,k,sub_size*l+sub_idx), 0.001);

                        float s0 = inp_h(i,j,k,sub_size*l+sub_idx);
                        float r  = res_h(  i  ,j,k,l);
                        float d  = delta(  i  ,j,k,l);
                        float f0 = d / (r) * s0;
                        BOOST_CHECK_CLOSE(1.f, 1.f + inp_grad_h(i,j,k,sub_size*l + sub_idx) - f0, 0.01f);
                    }

                }
            }
       std::cout << "test gradient norm(2) finished" << std::endl;
   }
   {
       std::cout << "in gradient max" << std::endl;
     using namespace cuv::alex_conv;
     unsigned int sub_size = 4;
     unsigned int nImg = 4;
     unsigned int nPix = 10;
     unsigned int nChan = 8 * sub_size;
     tensor<float,dev_memory_space,row_major> inp_grad(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> inp(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> res(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> delta(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     tensor<float,host_memory_space,row_major> inp_grad_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> res_h(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> delta_h(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     fill_rnd_uniform(inp_h);
     inp = inp_h;

     fill_rnd_uniform(res_h); // should be overwritten
     res = res_h; // copy to device
     res += 1.f;
     fill_rnd_uniform(inp_grad_h); // should be overwritten
     inp_grad = inp_grad_h; // copy to device
     inp_grad += 1.f;
     fill_rnd_uniform(delta_h);
     delta = delta_h;
     tuplewise_op(res,inp, 0, sub_size, TO_MAX);
     tuplewise_op(res_h,inp_h, 0, sub_size, TO_MAX);

     tuplewise_op_grad(inp_grad,inp,delta, 0, sub_size, TO_MAX);
     tuplewise_op_grad(inp_grad_h,inp_h,delta_h, 0, sub_size, TO_MAX);

     for(unsigned int i=0;i<nChan/sub_size;i++)
         for(unsigned int j=0;j<nPix;j++)
             for(unsigned int k=0;k<nPix;k++){
                 for(unsigned int l=0;l<nImg;l++){
                     float max_act = inp_h(sub_size*i,j,k,l);
                     unsigned int max_index = 0;
                     for (unsigned int sub_idx = 0; sub_idx < sub_size; sub_idx++){
                         BOOST_CHECK_CLOSE(1.f, 1.f + inp_grad(sub_size*i+sub_idx,j,k,l) - inp_grad_h(sub_size*i+sub_idx,j,k,l), 0.001);

                         float s0 = inp_h(sub_size*i+sub_idx,j,k,l);
                         if (s0 > max_act){
                             max_index = sub_idx; 
                             max_act = s0;
                         }
                     }
                     for (unsigned int sub_idx = 1; sub_idx < sub_size; sub_idx++){
                         if (sub_idx == max_index){
                             BOOST_CHECK_CLOSE(1.f + (float)delta(i,j,k,l), 1.f + (float)inp_grad_h(sub_size*i + sub_idx,j,k,l), 0.01f);
                         }else{
                             BOOST_CHECK_CLOSE(1.f, 1.f + inp_grad_h(sub_size*i + sub_idx,j,k,l), 0.01f);
                         }

                     }

                 }
             }
       std::cout << "test gradient max finished" << std::endl;
   }
   {
       std::cout << "in gradient max(2)" << std::endl;
     using namespace cuv::alex_conv;
     unsigned int sub_size = 3;
     unsigned int nImg = 2 * sub_size;
     unsigned int nPix = 2;
     unsigned int nChan = 2;
     tensor<float,dev_memory_space,row_major> inp_grad(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> inp(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> res(cuv::extents[nChan][nPix][nPix][nImg/sub_size]);
     tensor<float,dev_memory_space,row_major> delta(cuv::extents[nChan][nPix][nPix][nImg/sub_size]);

     tensor<float,host_memory_space,row_major> inp_grad_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> res_h(cuv::extents[nChan][nPix][nPix][nImg/sub_size]);
     tensor<float,host_memory_space,row_major> delta_h(cuv::extents[nChan][nPix][nPix][nImg/sub_size]);

     fill_rnd_uniform(inp_h);
     inp = inp_h;

     fill_rnd_uniform(delta_h);
     fill_rnd_uniform(res_h); // should be overwritten
     res = res_h; // copy to device
     res += 1.f;
     fill_rnd_uniform(inp_grad_h); // should be overwritten
     inp_grad = inp_grad_h; // copy to device
     inp_grad += 1.f;
     delta = delta_h;
     tuplewise_op(res,inp, 3, sub_size, TO_MAX, eps);
     tuplewise_op(res_h,inp_h, 3, sub_size,TO_MAX,eps);

     tuplewise_op_grad(inp_grad,inp,delta, 3, sub_size,TO_MAX, eps);
     tuplewise_op_grad(inp_grad_h,inp_h,delta_h, 3, sub_size,TO_MAX, eps);

     for(unsigned int i=0;i<nChan;i++)
         for(unsigned int j=0;j<nPix;j++)
             for(unsigned int k=0;k<nPix;k++){
                 for(unsigned int l=0;l<nImg/sub_size;l++){
                     for (unsigned int sub_idx = 0; sub_idx < sub_size; sub_idx++){
                         BOOST_CHECK_CLOSE(1.f, 1.f + inp_grad(i,j,k,sub_size*l+sub_idx) - inp_grad_h(i,j,k,sub_size*l+sub_idx), 0.001);

                         float s0 = inp_h(i,j,k,sub_size*l+sub_idx);
                         float r  = res_h(  i  ,j,k,l);
                         float d  = delta(  i  ,j,k,l);
                         float f0 = r == s0 ? d : 0.f;
                         BOOST_CHECK_CLOSE(1.f, 1.f + inp_grad_h(i,j,k,sub_size*l+sub_idx) - f0, 0.01f);
                     }

                 }
             }
       std::cout << "in gradient max(2) finished"<< std::endl;
   }
   // test for square norm
   {
     std::cout << "in sqr norm grad" << std::endl;/* cursor */
     using namespace cuv::alex_conv;
     unsigned int sub_size = 3;
     unsigned int nImg = 5;
     unsigned int nPix = 4;
     unsigned int nChan = 4 * sub_size;
     tensor<float,dev_memory_space,row_major> inp_grad(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> inp(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> res(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> delta(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     tensor<float,host_memory_space,row_major> inp_grad_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> res_h(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> delta_h(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     fill_rnd_uniform(inp_h);
     inp = inp_h;

     fill_rnd_uniform(res_h); // should be overwritten
     res = res_h; // copy to device
     fill_rnd_uniform(inp_grad_h); // should be overwritten
     inp_grad = inp_grad_h; // copy to device

     fill_rnd_uniform(delta_h);
     delta = delta_h;
     tuplewise_op(res,inp, 0, sub_size, TO_ADD_SQUARED, eps);
     tuplewise_op(res_h,inp_h, 0, sub_size, TO_ADD_SQUARED, eps);

     tuplewise_op_grad(inp_grad,inp,delta, 0, sub_size, TO_ADD_SQUARED, eps);
     tuplewise_op_grad(inp_grad_h,inp_h,delta_h, 0, sub_size, TO_ADD_SQUARED, eps);

     for(unsigned int i=0;i<nChan/sub_size;i++)
         for(unsigned int j=0;j<nPix;j++)
             for(unsigned int k=0;k<nPix;k++){
                 for(unsigned int l=0;l<nImg;l++){
                     for (unsigned int sub_idx = 0; sub_idx < sub_size; sub_idx++){
                         BOOST_CHECK_CLOSE(1.f, 1.f + inp_grad(sub_size*i+sub_idx,j,k,l) - inp_grad_h(sub_size*i+sub_idx,j,k,l), 0.001);

                         float s0 = 2*inp_h(sub_size*i+sub_idx,j,k,l);
                         float r  = res_h(  i  ,j,k,l);
                         float d  = delta(  i  ,j,k,l);
                         float f0 = d * s0;
                         BOOST_CHECK_CLOSE(1.f, 1.f + inp_grad_h(sub_size*i + sub_idx,j,k,l) - f0, 0.01f);
                     }

                 }
             }
     std::cout << "test sqr norm grad finished" << std::endl;/* cursor */
   }
   // test for subsample 
   {
     std::cout << "in gradient subsample" << std::endl;/* cursor */
     using namespace cuv::alex_conv;
     unsigned int sub_size = 3;
     unsigned int nImg = 5;
     unsigned int nPix = 4;
     unsigned int nChan = 4 * sub_size;
     tensor<float,dev_memory_space,row_major> inp_grad(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> inp(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> res(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> delta(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     tensor<float,host_memory_space,row_major> inp_grad_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> res_h(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> delta_h(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     fill_rnd_uniform(inp_h);
     inp = inp_h;

     fill_rnd_uniform(res_h);
     res = res_h;
     tuplewise_op(res,inp, 0, sub_size, TO_SUBSAMPLE, eps);
     tuplewise_op(res_h,inp_h, 0, sub_size, TO_SUBSAMPLE, eps);

     tuplewise_op_grad(inp_grad,inp,delta, 0, sub_size, TO_SUBSAMPLE, eps);
     tuplewise_op_grad(inp_grad_h,inp_h,delta_h, 0, sub_size, TO_SUBSAMPLE, eps);

     for(unsigned int i=0;i<nChan/sub_size;i++)
         for(unsigned int j=0;j<nPix;j++)
             for(unsigned int k=0;k<nPix;k++){
                 for(unsigned int l=0;l<nImg;l++){
                     for (unsigned int sub_idx = 0; sub_idx < sub_size; sub_idx++){
                         BOOST_CHECK_CLOSE(1.f, 1.f + inp_grad(sub_size*i+sub_idx,j,k,l) - inp_grad_h(sub_size*i+sub_idx,j,k,l), 0.001);

                         float r  = res_h(  i  ,j,k,l);
                         float d  = delta(  i  ,j,k,l);
                         float f0;
                         if(sub_idx == 0){
                             f0 = d;
                         }
                         else
                             f0 = 0.f;
                         BOOST_CHECK_CLOSE(1.f, 1.f + inp_grad_h(sub_size*i + sub_idx,j,k,l) - f0, 0.01f);
                     }

                 }
             }
     std::cout << "test subsample grad finished" << std::endl;/* cursor */
   }
   // test for mean 
   {
     std::cout << "in gradient mean" << std::endl;/* cursor */
     using namespace cuv::alex_conv;
     unsigned int sub_size = 3;
     unsigned int nImg = 5;
     unsigned int nPix = 4;
     unsigned int nChan = 4 * sub_size;
     tensor<float,dev_memory_space,row_major> inp_grad(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> inp(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> res(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);
     tensor<float,dev_memory_space,row_major> delta(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     tensor<float,host_memory_space,row_major> inp_grad_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nChan][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> res_h(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);
     tensor<float,host_memory_space,row_major> delta_h(cuv::extents[nChan/sub_size][nPix][nPix][nImg]);

     fill_rnd_uniform(inp_h);
     inp = inp_h;

     fill_rnd_uniform(res_h);
     res = res_h;
     fill_rnd_uniform(delta_h);
     delta = delta_h;
     tuplewise_op(res,inp, 0, sub_size, TO_MEAN, eps);
     tuplewise_op(res_h,inp_h, 0, sub_size, TO_MEAN, eps);

     tuplewise_op_grad(inp_grad,inp,delta, 0, sub_size, TO_MEAN, eps);
     tuplewise_op_grad(inp_grad_h,inp_h,delta_h, 0, sub_size, TO_MEAN, eps);

     for(unsigned int i=0;i<nChan/sub_size;i++)
         for(unsigned int j=0;j<nPix;j++)
             for(unsigned int k=0;k<nPix;k++){
                 for(unsigned int l=0;l<nImg;l++){
                     for (unsigned int sub_idx = 0; sub_idx < sub_size; sub_idx++){
                         BOOST_CHECK_CLOSE(1.f, 1.f + inp_grad(sub_size*i+sub_idx,j,k,l) - inp_grad_h(sub_size*i+sub_idx,j,k,l), 0.001);

                         float s0 = inp_h(sub_size*i+sub_idx,j,k,l);
                         float r  = res_h(  i  ,j,k,l);
                         float d  = delta(  i  ,j,k,l);
                         float f0;
                         f0 = d * (1.f / sub_size);
                         BOOST_CHECK_CLOSE(1.f, 1.f + inp_grad_h(sub_size*i + sub_idx,j,k,l) - f0, 0.01f);
                     }

                 }
             }
     std::cout << "test mean grad finished" << std::endl;/* cursor */
   }
   // test for mean 2 
   {
     std::cout << "in gradient mean 2" << std::endl;/* cursor */
     using namespace cuv::alex_conv;
     unsigned int sub_size = 3;
     unsigned int nImg = 5;
     unsigned int nPix = 4;
     unsigned int nChan = 4 * sub_size;
     tensor<float,dev_memory_space,row_major> inp_grad(cuv::extents[nImg][nPix][nPix][nChan]);
     tensor<float,dev_memory_space,row_major> inp(cuv::extents[nImg][nPix][nPix][nChan]);
     tensor<float,dev_memory_space,row_major> res(cuv::extents[nImg][nPix][nPix][nChan/sub_size]);
     tensor<float,dev_memory_space,row_major> delta(cuv::extents[nImg][nPix][nPix][nChan/sub_size]);

     tensor<float,host_memory_space,row_major> inp_grad_h(cuv::extents[nImg][nPix][nPix][nChan]);
     tensor<float,host_memory_space,row_major> inp_h(cuv::extents[nImg][nPix][nPix][nChan]);
     tensor<float,host_memory_space,row_major> res_h(cuv::extents[nImg][nPix][nPix][nChan/sub_size]);
     tensor<float,host_memory_space,row_major> delta_h(cuv::extents[nImg][nPix][nPix][nChan/sub_size]);

     fill_rnd_uniform(inp_h);
     inp = inp_h;

     fill_rnd_uniform(res_h);
     res = res_h;
     fill_rnd_uniform(delta_h);
     delta = delta_h;
     tuplewise_op(res,inp, 3, sub_size, TO_MEAN, eps);
     tuplewise_op(res_h,inp_h, 3, sub_size, TO_MEAN, eps);

     tuplewise_op_grad(inp_grad,inp,delta, 3, sub_size, TO_MEAN, eps);
     tuplewise_op_grad(inp_grad_h,inp_h,delta_h, 3, sub_size, TO_MEAN, eps);

     for(unsigned int i=0;i<nImg;i++)
         for(unsigned int j=0;j<nPix;j++)
             for(unsigned int k=0;k<nPix;k++){
                 for(unsigned int l=0;l<nChan/sub_size;l++){
                     for (unsigned int sub_idx = 0; sub_idx < sub_size; sub_idx++){
                         BOOST_CHECK_CLOSE(1.f, 1.f + inp_grad(i,j,k,sub_size*l+sub_idx) - inp_grad_h(i,j,k,sub_size*l+sub_idx), 0.001);

                         float s0 = inp_h(i,j,k,sub_size*l+sub_idx);
                         float r  = res_h(  i  ,j,k,l);
                         float d  = delta(  i  ,j,k,l);
                         float f0;
                         f0 = d * (1.f / sub_size);
                         BOOST_CHECK_CLOSE(1.f, 1.f + inp_grad(i,j,k,sub_size*l+sub_idx) - f0, 0.01f);
                     }

                 }
             }
     std::cout << "test mean grad finished" << std::endl;/* cursor */
   }
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_CASE( test_upscale )    
{
   const int FACTOR    = 4;
   const int CHANNELS  = 4;
   const int HEIGHT    = 10;
   const int WIDTH     = 10;
   const int NR_IMAGES = 3;
   // forward test
   tensor<float,host_memory_space,row_major> testSetImages_h(cuv::extents[CHANNELS][HEIGHT][WIDTH][NR_IMAGES]);
   tensor<float,dev_memory_space,row_major>  testSetImages_d(cuv::extents[CHANNELS][HEIGHT][WIDTH][NR_IMAGES]);

   tensor<float,host_memory_space,row_major>   upscaledImages_h(cuv::extents[CHANNELS][HEIGHT*FACTOR][WIDTH*FACTOR][NR_IMAGES]);
   tensor<float,dev_memory_space,row_major> upscaledImages_d(cuv::extents[CHANNELS][HEIGHT*FACTOR][WIDTH*FACTOR][NR_IMAGES]);

   // generate a dummy set of images
	for (int i = 0; i < CHANNELS; i++)
		for (int x = 0; x < HEIGHT; x++)
			for (int y = 0; y < WIDTH; y++)
				for (int k = 0; k < NR_IMAGES; k++)
            {
					testSetImages_h(i,x,y,k) = rand() % 500;
					testSetImages_d(i,x,y,k) = testSetImages_h(i,x,y,k);
            }

   MEASURE_TIME(upscaleOp_dev, cuv::misc_conv::upscaleOp(upscaledImages_d, testSetImages_d, FACTOR), 100);

   MEASURE_TIME(upscaleOp_host,cuv::misc_conv::upscaleOp(upscaledImages_h, testSetImages_h, FACTOR), 100);


   // check if they are the same
   for (int i = 0; i < CHANNELS; i++)
		   for (int x = 0; x < HEIGHT*FACTOR; x++)
			   for (int y = 0; y < WIDTH*FACTOR; y++)
				   for (int k = 0; k < NR_IMAGES; k++)
               {
                  float v1 = upscaledImages_h(i,x,y,k);
                  float v2 = upscaledImages_d(i,x,y,k);
					   BOOST_CHECK_CLOSE(v1 , v2 , 0.0001f); 
               }
   // backward test

   // generate a dummy set of gradients
	for (int i = 0; i < CHANNELS; i++)
		for (int x = 0; x < HEIGHT*FACTOR; x++)
			for (int y = 0; y < WIDTH*FACTOR; y++)
				for (int k = 0; k < NR_IMAGES; k++)
            {
					upscaledImages_h(i,x,y,k) = rand() % 500;
					upscaledImages_d(i,x,y,k) = upscaledImages_h(i,x,y,k);
            }

   MEASURE_TIME(upscaleGrad_dev, cuv::misc_conv::upscaleGrad(testSetImages_d, upscaledImages_d, FACTOR), 100);
   MEASURE_TIME(upscaleGrad_host,cuv::misc_conv::upscaleGrad(testSetImages_h, upscaledImages_h, FACTOR), 100);


   // check if they are the same
   for (int i = 0; i < CHANNELS; i++)
		   for (int x = 0; x < HEIGHT; x++)
			   for (int y = 0; y < WIDTH; y++)
				   for (int k = 0; k < NR_IMAGES; k++)
               {
                  float v1 = testSetImages_h(i,x,y,k);
                  float v2 = testSetImages_d(i,x,y,k);
					   BOOST_CHECK_CLOSE(v1 , v2 , 0.0001f );
               }

}
