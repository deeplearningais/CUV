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
#include <numeric>
#include <boost/test/included/unit_test.hpp>
#include <cuv/tools/cuv_test.hpp>
#include <cuv/tools/timing.hpp>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/tensor.hpp>
#include <cuv/libs/theano_ops/theano_ops.hpp>

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
	}
	~Fix(){
	}
};

BOOST_FIXTURE_TEST_SUITE( s, Fix )

BOOST_AUTO_TEST_CASE( test_dim_shuffle )
{
    using namespace cuv::theano_ops;
    initcuda();
    {
      unsigned int nImg = 3;
      unsigned int npix = 6;

      cuv::tensor<float,cuv::dev_memory_space> src(cuv::extents[nImg][npix]);
      cuv::tensor<float,cuv::dev_memory_space> dst(cuv::extents[nImg][npix]);

      for (int i = 0; i < nImg; ++i)
      {
          for (int y = 0; y < npix; ++y)
          {
              src(i,y) = i * y + y + i;
          }
      }
      dst = 1.f;

      dim_shuffle(dst, src, cuv::extents[1][0]);
      BOOST_CHECK_EQUAL(dst.shape(1), src.shape(0));
      BOOST_CHECK_EQUAL(dst.shape(0), src.shape(1));


      for (int i = 0; i < nImg; ++i)
      {
          for (int y = 0; y < npix; ++y)
          {
              BOOST_CHECK_EQUAL(dst(y,i), src(i, y));
          }
      }
    }
    {
      unsigned int nImg = 2;
      unsigned int nChan = 3;
      unsigned int npix = 4;

      cuv::tensor<float,cuv::dev_memory_space> src(cuv::extents[nImg][nChan][npix]);
      cuv::tensor<float,cuv::dev_memory_space> dst(cuv::extents[nImg][nChan][npix]);

      for (int i = 0; i < nImg; ++i)
      {
          for (int c = 0; c < nChan; ++c){
              for (int y = 0; y < npix; ++y)
              {
                  src(i,c,y) = i * npix * nChan + c * npix + y;
              }
          }
      }
      dst = 1.f;

      dim_shuffle(dst,src, cuv::extents[1][0][2]);
      BOOST_CHECK_EQUAL(dst.shape(0), src.shape(1));
      BOOST_CHECK_EQUAL(dst.shape(1), src.shape(0));
      BOOST_CHECK_EQUAL(dst.shape(2), src.shape(2));


      for (int i = 0; i < nImg; ++i)
      {
          for (int c = 0; c < nChan; ++c){
              for (int y = 0; y < npix; ++y)
              {
                  BOOST_CHECK_EQUAL(dst(c,i, y), src(i, c, y));
              }

          }
      }
    }
{
   unsigned int nImg = 10;
   unsigned int nChan = 5;
   unsigned int npix_x = 1;
   unsigned int npix_y = 10;

   cuv::tensor<float,cuv::dev_memory_space> src(cuv::extents[nImg][nChan][npix_x][npix_y]);
   cuv::tensor<float,cuv::dev_memory_space> dst(cuv::extents[nImg][nChan][npix_x][npix_y]);

   for (int i = 0; i < nImg; ++i)
   {
       for (int c = 0; c < nChan; ++c)
       {
           for (int x = 0; x < npix_x; ++x)
           {
               for (int y = 0; y < npix_y; ++y)
               {
                       src(i,c,x,y) = i* nChan * npix_x * npix_y + c* npix_x * npix_y + x*npix_y + y;
               }
           }
       }
   }
   dst = 1.f;

   dim_shuffle(dst, src, cuv::extents[1][0][2][3]);
   BOOST_CHECK_EQUAL(dst.shape(1), src.shape(0));
   BOOST_CHECK_EQUAL(dst.shape(0), src.shape(1));
   BOOST_CHECK_EQUAL(dst.shape(2), src.shape(2));
   BOOST_CHECK_EQUAL(dst.shape(3), src.shape(3));


   for (int i = 0; i < nImg; ++i)
   {
       for (int c = 0; c < nChan; ++c)
       {
           for (int x = 0; x < npix_x; ++x)
           {
               for (int y = 0; y < npix_y; ++y)
               {
                   BOOST_CHECK_EQUAL(dst(c,i,x,y), src(i,c,x,y));
               }
           }
       }
   }
   std::cout << " 1 dim tensor finished " << std::endl;/* cursor */ 
}
{
   unsigned int nImg = 3;
   unsigned int nChan = 4;
   unsigned int npix_x = 2;
   unsigned int npix_y = 4;

   cuv::tensor<float,cuv::dev_memory_space> src(cuv::extents[nImg][nChan][npix_x][npix_y]);
   cuv::tensor<float,cuv::dev_memory_space> dst(cuv::extents[nImg][nChan][npix_x][npix_y]);

   for (int i = 0; i < nImg; ++i)
   {
       for (int c = 0; c < nChan; ++c)
       {
           for (int x = 0; x < npix_x; ++x)
           {
               for (int y = 0; y < npix_y; ++y)
               {
                       src(i,c,x,y) = i* nChan * npix_x * npix_y + c* npix_x * npix_y + x*npix_y + y;
               }
           }
       }
   }
   dst = 1.f;

   dim_shuffle(dst, src, cuv::extents[0][1][3][2]);
   BOOST_CHECK_EQUAL(dst.shape(0), src.shape(0));
   BOOST_CHECK_EQUAL(dst.shape(1), src.shape(1));
   BOOST_CHECK_EQUAL(dst.shape(3), src.shape(2));
   BOOST_CHECK_EQUAL(dst.shape(2), src.shape(3));


   for (int i = 0; i < nImg; ++i)
   {
       for (int c = 0; c < nChan; ++c)
       {
           for (int x = 0; x < npix_x; ++x)
           {
               for (int y = 0; y < npix_y; ++y)
               {
                   BOOST_CHECK_EQUAL(dst(i,c,y,x), src(i,c,x,y));
               }
           }
       }
   }
    
}

{
   unsigned int nImg = 3;
   unsigned int nChan = 4;
   unsigned int npix_x = 2;
   unsigned int npix_y = 4;

   cuv::tensor<float,cuv::dev_memory_space> src(cuv::extents[nImg][nChan][npix_x][npix_y]);
   cuv::tensor<float,cuv::dev_memory_space> dst(cuv::extents[nImg][nChan][npix_x][npix_y]);

   for (int i = 0; i < nImg; ++i)
   {
       for (int c = 0; c < nChan; ++c)
       {
           for (int x = 0; x < npix_x; ++x)
           {
               for (int y = 0; y < npix_y; ++y)
               {
                       src(i,c,x,y) = i* nChan * npix_x * npix_y + c* npix_x * npix_y + x*npix_y + y;
               }
           }
       }
   }
   dst = 1.f;

   dim_shuffle(dst, src, cuv::extents[0][3][2][1]);
   BOOST_CHECK_EQUAL(dst.shape(0), src.shape(0));
   BOOST_CHECK_EQUAL(dst.shape(3), src.shape(1));
   BOOST_CHECK_EQUAL(dst.shape(2), src.shape(2));
   BOOST_CHECK_EQUAL(dst.shape(1), src.shape(3));


   for (int i = 0; i < nImg; ++i)
   {
       for (int c = 0; c < nChan; ++c)
       {
           for (int x = 0; x < npix_x; ++x)
           {
               for (int y = 0; y < npix_y; ++y)
               {
                   BOOST_CHECK_EQUAL(dst(i,y,x,c), src(i,c,x,y));
               }
           }
       }
   }
    
}
    finalize_cuda();
}

BOOST_AUTO_TEST_CASE( test_flip_dims )
{
    using namespace cuv::theano_ops;
    {
        unsigned int nImg = 2;
        unsigned int nChan = 2;
        unsigned int npix_x = 1;
        unsigned int npix_y = 4;

        initcuda();
        cuv::tensor<float,cuv::dev_memory_space> src(cuv::extents[nImg][nChan][npix_x][npix_y]);
        cuv::tensor<float,cuv::dev_memory_space> dst(cuv::extents[nImg][nChan][npix_x][npix_y]);

        for (int i = 0; i < nImg; ++i)
        {
            for (int c = 0; c < nChan; ++c)
            {
                for (int x = 0; x < npix_x; ++x)
                {
                    for (int y = 0; y < npix_y; ++y)
                    {
                        src(i,c,x,y) = i* nChan * npix_x * npix_y + c* npix_x * npix_y + x*npix_y + y;
                    }
                }
            }
        }
        dst = 1.f;

        {   
            flip_dims(dst, src, cuv::extents[0][0][1][1]);

            BOOST_CHECK_EQUAL(dst.shape(0), src.shape(0));
            BOOST_CHECK_EQUAL(dst.shape(1), src.shape(1));
            BOOST_CHECK_EQUAL(dst.shape(2), src.shape(2));
            BOOST_CHECK_EQUAL(dst.shape(3), src.shape(3));

            for (int i = 0; i < nImg; ++i)
            {
                for (int c = 0; c < nChan; ++c)
                {
                    for (int x = 0; x < npix_x; ++x)
                    {
                        for (int y = 0; y < npix_y; ++y)
                        {
                            BOOST_CHECK_EQUAL(dst(i,c, npix_x - 1 - x, npix_y - 1 - y ), src(i,c,x,y));
                        }
                    }
                }
            }

        }
        {   
            flip_dims(dst, src, cuv::extents[1][1][0][0]);

            BOOST_CHECK_EQUAL(dst.shape(0), src.shape(0));
            BOOST_CHECK_EQUAL(dst.shape(1), src.shape(1));
            BOOST_CHECK_EQUAL(dst.shape(2), src.shape(2));
            BOOST_CHECK_EQUAL(dst.shape(3), src.shape(3));

            for (int i = 0; i < nImg; ++i)
            {
                for (int c = 0; c < nChan; ++c)
                {
                    for (int x = 0; x < npix_x; ++x)
                    {
                        for (int y = 0; y < npix_y; ++y)
                        {
                            BOOST_CHECK_EQUAL(dst(nImg-1-i,nChan-1-c, x, y ), src(i,c,x,y));
                        }
                    }
                }
            }

        }
        {   
            std::vector<bool> v(4);
            v[0] = true;
            v[1] = true;
            v[2] = true;
            v[3] = true;
            flip_dims_vec(dst, src, v);

            BOOST_CHECK_EQUAL(dst.shape(0), src.shape(0));
            BOOST_CHECK_EQUAL(dst.shape(1), src.shape(1));
            BOOST_CHECK_EQUAL(dst.shape(2), src.shape(2));
            BOOST_CHECK_EQUAL(dst.shape(3), src.shape(3));

            for (int i = 0; i < nImg; ++i)
            {
                for (int c = 0; c < nChan; ++c)
                {
                    for (int x = 0; x < npix_x; ++x)
                    {
                        for (int y = 0; y < npix_y; ++y)
                        {
                            BOOST_CHECK_EQUAL(dst(nImg-1-i,nChan-1-c, npix_x - 1 - x, npix_y - 1 - y ), src(i,c,x,y));
                        }
                    }
                }
            }

        }

    }
    finalize_cuda();
}









BOOST_AUTO_TEST_SUITE_END()
