cmake_minimum_required( VERSION 2.6 FATAL_ERROR )

OPTION(CUV_PYTHON_BINDINGS "Whether shared version of CUDA is to be used, must be true for python bindings!" ON)
IF(CUV_PYTHON_BINDINGS)
#	use this for python bindings. However, We need to recompile CUDA for this on every computer!
	SET(CMAKE_CXX_FLAGS "-fPIC") # for FindCUDA, needed for python shared lib
ENDIF(CUV_PYTHON_BINDINGS)

SET(CUDA_SDK_ROOT_DIR "/usr/local/cuda/C" CACHE STRING "Location of CUDA SDK") 

SET (CUDA_TEST_DEVICE "3" CACHE STRING "Which CUDA device should the tests be run on, if multiple devices are present.")
add_definitions(-DCUDA_TEST_DEVICE=${CUDA_TEST_DEVICE})

FIND_PACKAGE(CUDA)

# ---------- Find Boost Headers/Libraries -----------------------
SET (Boost_FIND_REQUIRED TRUE)
SET (Boost_FIND_QUIETLY TRUE)
set (Boost_USE_MULTITHREADED TRUE)
set (Boost_USE_STATIC_LIBS FALSE)
SET (Boost_ADDITIONAL_VERSIONS "1.39" "1.39.0")
FIND_PACKAGE( Boost 1.37 COMPONENTS date_time unit_test_framework serialization REQUIRED )
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

add_definitions(-DRANDOM_PATH=${CMAKE_BINARY_DIR})
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-options;-DRANDOM_PATH=\\"${CMAKE_BINARY_DIR}\\")

CUDA_INCLUDE_DIRECTORIES( ${CUDA_SDK_ROOT_DIR}/common/inc ${CMAKE_CURRENT_SOURCE_DIR}  tools )
INCLUDE_DIRECTORIES(      ${CUDA_SDK_ROOT_DIR}/common/inc ${CUDA_INCLUDE_DIRS}         tools )
CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}                                )
INCLUDE_DIRECTORIES(      ${CMAKE_CURRENT_SOURCE_DIR}                                )

add_subdirectory(3rd_party)

add_subdirectory(tools)
add_subdirectory(basics)
add_subdirectory(vector_ops)
add_subdirectory(matrix_ops)
add_subdirectory(image_ops)
add_subdirectory(convolution_ops)
add_subdirectory(convert)
add_subdirectory(random)
add_subdirectory(libs)

add_subdirectory(tests)

IF(CUV_PYTHON_BINDINGS)
	add_subdirectory(python_bindings)
ENDIF(CUV_PYTHON_BINDINGS)


CUDA_BUILD_CLEAN_TARGET()


