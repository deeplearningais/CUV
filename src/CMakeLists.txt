cmake_minimum_required( VERSION 2.6 FATAL_ERROR )

PROJECT(CUV Fortran CXX C)
SET(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/CMakeModules )

INCLUDE(InstallRequiredSystemLibraries)

SET(CPACK_SOURCE_GENERATOR "TGZ")
SET(CPACK_PACKAGE_NAME "cuv")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Matrix Algorithms on GPU")
SET(CPACK_PACKAGE_CONTACT "Hannes Schulz <schulz at ais dot uni-bonn dot de>")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../README")
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "4")
SET(CPACK_PACKAGE_VERSION_PATCH "0")
SET(CPACK_SOURCE_IGNORE_FILES "${CPACK_SOURCE_IGNORE_FILES};cfg/config.h")
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "CMake ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}")
SET(CPACK_DEBIAN_PACKAGE_DEPENDS "libboost-dev (>= 1.34.1)" )
SET(CMAKE_INSTALL_PREFIX ".")
INCLUDE(CPack)

FIND_PACKAGE(Doxygen)
IF(DOXYGEN_FOUND)
	CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/../doc/Doxyfile.in ${CMAKE_BINARY_DIR}/doc/Doxyfile)
	ADD_CUSTOM_COMMAND(
		DEPENDS ${CMAKE_BINARY_DIR}/doc/Doxyfile
		OUTPUT  ${CMAKE_BINARY_DIR}/doc/html/index.html
		COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/doc/Doxyfile
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		)
	ADD_CUSTOM_TARGET(doc DEPENDS ${CMAKE_BINARY_DIR}/doc/html/index.html)
ENDIF(DOXYGEN_FOUND)

OPTION(CUDA_SHARED "Wether shared version of CUDA is to be used, must be true for python bindings!" ON)
IF(CUDA_SHARED)
#	use this for python bindings. However, We need to recompile CUDA for this on every computer!
	SET(CMAKE_CXX_FLAGS "-fPIC") # for FindCUDA, needed for python shared lib
	SET(CUDA_SDK_ROOT_DIR "/usr/local/src/cuda_shared" CACHE STRING "Location of CUDA SDK") 
ELSE(CUDA_SHARED)
	SET(CUDA_SDK_ROOT_DIR "/usr/local/src/cuda" CACHE STRING "Location of CUDA SDK")
ENDIF(CUDA_SHARED)

SET (CUDA_TEST_DEVICE "0" CACHE STRING "Which CUDA device should the tests be run on, if multiple devices are present.")
add_definitions(-DCUDA_TEST_DEVICE=${CUDA_TEST_DEVICE})

FIND_PACKAGE(CUDA)
#FIND_PACKAGE(CxxTest)


# ---------- Find Boost Headers/Libraries -----------------------
SET (Boost_FIND_REQUIRED TRUE)
SET (Boost_FIND_QUIETLY TRUE)
set (Boost_USE_MULTITHREADED TRUE)
set (Boost_USE_STATIC_LIBS FALSE)
SET (Boost_ADDITIONAL_VERSIONS "1.39" "1.39.0")
FIND_PACKAGE( Boost 1.37 COMPONENTS date_time unit_test_framework serialization REQUIRED )
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

add_definitions(-DRANDOM_PATH="\"${CMAKE_BINARY_DIR}\"")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-options;-DRANDOM_PATH=\\"${CMAKE_BINARY_DIR}\\")

CUDA_INCLUDE_DIRECTORIES( ${CUDA_CUT_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}  tools )
INCLUDE_DIRECTORIES(      ${CUDA_CUT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS}         tools )
CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}                                )
INCLUDE_DIRECTORIES(      ${CMAKE_CURRENT_SOURCE_DIR}                                )

ENABLE_TESTING()
add_subdirectory(3rd_party)

add_subdirectory(tools)
add_subdirectory(basics)
add_subdirectory(vector_ops)
add_subdirectory(matrix_ops)
add_subdirectory(convolution_ops)
add_subdirectory(convert)
add_subdirectory(random)

add_subdirectory(tests)

IF(CUDA_SHARED)
	add_subdirectory(python_bindings)
ENDIF(CUDA_SHARED)


CUDA_BUILD_CLEAN_TARGET()


