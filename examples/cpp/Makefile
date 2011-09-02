CUV_PATH          = /usr/include
CUV_LIBRARY_PATH  = /usr/lib
CUDA_LIBRARY_PATH = /usr/lib/nvidia-current
CUDA_TOOLKIT_LIB  = /usr/local/cuda/lib

INCLUDES = -I$(CUV_PATH)/src
LIB_PATH = -L$(CUDA_LIBRARY_PATH) -L$(CUV_LIBRARY_PATH) -L$(CUDA_LIBRARY_PATH) -L$(CUDA_TOOLKIT_LIB)
LIBS     = -lcuv -lcuda -lcudart -lcblas -lboost_serialization


all: example1 example2 serialization
example1: example1.cpp
	gcc example1.cpp $(INCLUDES) $(LIB_PATH) $(LIBS) -o example1
example2: example2.cpp
	gcc example2.cpp $(INCLUDES) $(LIB_PATH) $(LIBS) -o example2
serialization: serialization.cpp
	gcc -g3 -O0 serialization.cpp $(INCLUDES) $(LIB_PATH) $(LIBS) -o serialization
clean:
	rm -f *.o example1 example2 serialization
