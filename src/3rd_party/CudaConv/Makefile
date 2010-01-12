COMMONFLAGS := -DRND_MULTIPLIERS_FILE=\"rnd_multipliers_32bit.txt\"
ATLAS_LIB:=/usr/lib/atlas
INCLUDES := 
LIB := -L$(ATLAS_LIB) -latlas -lcblas

EXECUTABLE	:= conv

USECUBLAS   := 1
# Add source files here

# CUDA source files (compiled with cudacc)
CUFILES		:= testconv.cu conv.cu conv2.cu conv3.cu testconv_extras.cu conv_util.cu nvmatrix.cu nvmatrix_kernel.cu
# CUDA dependency files
CU_DEPS		:= conv.cuh conv2.cuh conv3.cuh testconv_extras.cuh conv_util.cuh nvmatrix.cuh nvmatrix_kernel.cuh

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= convCPU.cpp gpu_locking.cpp matrix.cpp
C_DEPS		:= convCPU.h gpu_locking.h matrix.h matrix_funcs.h


################################################################################
# Rules and targets

ifeq ($(CUDA_VERSION), 2.3)
	include common-gcc-cuda-2.3.mk
else
	include common-gcc-cuda-2.1.mk
endif
