FIND_PACKAGE( BLAS )


CUDA_INCLUDE_DIRECTORIES( ${CBLAS_INCLUDE_DIRECTORY} )
CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} )
#CUDA_INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/basics )
#CUDA_INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/convert )
#CUDA_INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/vector_ops )
#CUDA_INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/matrix_ops )

INCLUDE_DIRECTORIES( ${CBLAS_INCLUDE_DIRECTORY} )
#INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} )
#INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/basics )
#INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/convert )
#INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/vector_ops )
#INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/matrix_ops )

CUDA_ADD_LIBRARY(cuda_conv STATIC
#	testconv.cu
	conv.cu
	conv2.cu
	conv3.cu
#	testconv_extras.cu
	conv_util.cu
	nvmatrix.cu
	nvmatrix_kernel.cu

#	conv.cuh
#	conv2.cuh
#	conv3.cuh
#	testconv_extras.cuh
#	conv_util.cuh
#	nvmatrix.cuh
#	nvmatrix_kernel.cuh

	convCPU.cpp
	gpu_locking.cpp
	matrix.cpp

#	convCPU.h
#	gpu_locking.h
#	matrix.h
#	matrix_funcs.h
  )

TARGET_LINK_LIBRARIES(cuda_conv ${CUDA_LIBRARIES} ${CUDA_CUT_LIBRARY} ${BLAS_LIBRARIES} )

install(TARGETS cuda_conv
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib/static)





#cmake_minimum_required(VERSION 2.6)
#project(CUDACONV)

#SET(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/../CMakeModules )
#SET(CUDA_SDK_ROOT_DIR "/usr/local/src/CUDA/C" CACHE STRING "Location of CUDA SDK")

#FIND_PACKAGE(CUDA)

#CUDA_INCLUDE_DIRECTORIES(
#    ${CUDA_CUT_INCLUDE_DIR}
#    ${CUDA_PATH}
#    ${CUDA_PATH}/projects
#    ${CMAKE_CURRENT_SOURCE_DIR}
#)

#INCLUDE_DIRECTORIES(
#    ${CUDA_CUT_INCLUDE_DIR}
#    ${CUDA_INCLUDE_DIR}
#    ${CUDA_TOOLKIT_INCLUDE}
#)

#CUDA_ADD_LIBRARY(cudaconv
#	testconv.cu
#	conv.cu
#	conv2.cu
#	conv3.cu
#	testconv_extras.cu
#	conv_util.cu
#	nvmatrix.cu
#	nvmatrix_kernel.cu
#
#	conv.cuh
#	conv2.cuh
#	conv3.cuh
#	testconv_extras.cuh
#	conv_util.cuh
#	nvmatrix.cuh
#	nvmatrix_kernel.cuh
#
#	convCPU.cpp
#	gpu_locking.cpp
#	matrix.cpp
#
#	convCPU.h
#	gpu_locking.h
#	matrix.h
#	matrix_funcs.h
#)

