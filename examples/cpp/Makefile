CUV_PATH          = /usr/include
CUV_LIBRARY_PATH  = /usr/lib
CUDA_LIBRARY_PATH = /usr/lib/nvidia-current
CUDA_TOOLKIT_LIB  = /usr/local/cuda/lib

INCLUDES = -I$(CUV_PATH)/src
LIB_PATH = -L$(CUDA_LIBRARY_PATH) -L$(CUV_LIBRARY_PATH) -L$(CUDA_LIBRARY_PATH) -L$(CUDA_TOOLKIT_LIB)
LIBS     = -lcuv -lcuda -lcudart -lcblas


all: example1 example2
example1:	
	gcc example1.cpp $(INCLUDES) $(LIB_PATH) $(LIBS) -o example1
example2:
	gcc example2.cpp $(INCLUDES) $(LIB_PATH) $(LIBS) -o example2

