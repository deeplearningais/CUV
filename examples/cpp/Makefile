CUV_PATH = /home/data/amueller/checkout/cuv
CUV_LIBRARY_PATH = /home/data/amueller/checkout/cuv/build/lib/static
CUDA_LIBRARY_PATH = /usr/lib/nvidia-current
CUDA_TOOLKIT_LIB = /usr/local/cuda/lib

INCLUDES = -I$(CUV_PATH)/src
LIB_PATH = -L$(CUDA_LIBRARY_PATH) -L$(CUV_LIBRARY_PATH) -L$(CUDA_LIBRARY_PATH) -L$(CUDA_TOOLKIT_LIB)
LIBS = -lcuv_matrix_ops -lcublas -lcuv_tensor_ops -lcuv_basics -lcuv_tools -lcuda -lcudart -lcblas


example1:	
	gcc example1.cpp $(INCLUDES) $(LIB_PATH) $(LIBS) -o example1
example2:
	gcc example2.cpp $(INCLUDES) $(LIB_PATH) $(LIBS) -o example2
all:
	example1

