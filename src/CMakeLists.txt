cmake_minimum_required( VERSION 2.6 FATAL_ERROR )

OPTION(CUV_PYTHON_BINDINGS "Whether shared version of CUDA is to be used, must be true for python bindings!" ON)
OPTION(CUV_CIMG_BINDINGS "Whether to compile with CImg" ON)
IF(CUV_PYTHON_BINDINGS)
#	use this for python bindings. However, We need to recompile CUDA for this on every computer!
	SET(CMAKE_CXX_FLAGS "-fPIC") # for FindCUDA, needed for python shared lib
ENDIF(CUV_PYTHON_BINDINGS)

SET (CUDA_SDK_ROOT_DIR "/usr/local/cuda/C" CACHE STRING "Location of CUDA SDK (currently not used) ") 

FIND_PACKAGE(CUDA)
#SET (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_21")

option (CUDA_ENFORCE_HOST_COMPILER "Force nvcc to use the same compiler used to compile .c(pp) files insted of gcc/g++" ON)
if (CUDA_ENFORCE_HOST_COMPILER)
    set (CMAKE_GCC_TEMP_DIR "${CMAKE_BINARY_DIR}/CMakeGCC")
    file(MAKE_DIRECTORY ${CMAKE_GCC_TEMP_DIR})
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_C_COMPILER} ${CMAKE_GCC_TEMP_DIR}/gcc)
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CXX_COMPILER} ${CMAKE_GCC_TEMP_DIR}/g++)
    set(CUDA_NVCC_FLAGS -ccbin ${CMAKE_GCC_TEMP_DIR} ${CUDA_NVCC_FLAGS})
endif()

if ( NOT CUDA_ARCH )
    SET( CUDA_ARCH -gencode;arch=compute_13,code=sm_13;-gencode;arch=compute_20,code=sm_20;-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35  CACHE STRING "Device architectures")
    #SET( CUDA_ARCH -gencode;arch=compute_20,code=\"sm_20,compute_20\";-gencode;arch=compute_30,code=\"sm_30,compute_30\";-gencode;arch=compute_35,code=\"sm_35,compute_35\" )
endif()

SET( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH} )

# ---------- Find Boost Headers/Libraries -----------------------
SET (Boost_FIND_REQUIRED TRUE)
SET (Boost_FIND_QUIETLY TRUE)
SET (Boost_USE_MULTITHREADED TRUE)
SET (Boost_USE_STATIC_LIBS FALSE)
SET (Boost_ADDITIONAL_VERSIONS "1.39" "1.39.0" "1.40" "1.42" "1.43" "1.44" "1.45" "1.46" "1.47" "1.48")
FIND_PACKAGE( Boost 1.37 COMPONENTS date_time unit_test_framework serialization thread system REQUIRED )
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

FIND_PATH(THRUST_PATH thrust/device_vector.h /usr/include /usr/local/include ${CUDA_INCLUDE_DIRS} "$ENV{THRUST_ROOT}")
IF(NOT THRUST_PATH)
	MESSAGE(FATAL_ERROR "Could not find the thrust library. Please install in standard locations or set THRUST_ROOT environment variable.")
ENDIF(NOT THRUST_PATH)

add_definitions(-DRANDOM_PATH=${CMAKE_BINARY_DIR})
#add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
SET(CUDA_ARCHITECTURE "" CACHE STRING "The CUDA architecture to compile for, i.e. -arch=sm_20")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CUDA_ARCHITECTURE}")
MESSAGE(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

CUDA_INCLUDE_DIRECTORIES( ${CUDA_SDK_ROOT_DIR}/common/inc ${CMAKE_CURRENT_SOURCE_DIR}  tools )
INCLUDE_DIRECTORIES(      ${CUDA_SDK_ROOT_DIR}/common/inc ${CUDA_INCLUDE_DIRS}         tools )
CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}                                )
INCLUDE_DIRECTORIES(      ${CMAKE_CURRENT_SOURCE_DIR}                                )

CUDA_INCLUDE_DIRECTORIES( ${THRUST_PATH}                                )
INCLUDE_DIRECTORIES(      ${THRUST_PATH}                                )

add_subdirectory(3rd_party)
add_subdirectory(cuv)
add_subdirectory(tests EXCLUDE_FROM_ALL)

IF(CUV_PYTHON_BINDINGS)
	add_subdirectory(cuv_python)
ENDIF(CUV_PYTHON_BINDINGS)

INSTALL(DIRECTORY cuv${LIB_SUFFIX}/
	DESTINATION "include/cuv"
	FILES_MATCHING PATTERN "*.hpp"
)
IF("${LIB_SUFFIX}" STREQUAL "")
    INSTALL(FILES cuv.hpp DESTINATION include)
ENDIF("${LIB_SUFFIX}" STREQUAL "")

CUDA_BUILD_CLEAN_TARGET()


