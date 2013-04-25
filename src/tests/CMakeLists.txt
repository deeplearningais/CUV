INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src/basics )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src/convert )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src/tensor_ops )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src/matrix_ops )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src/convolution_ops )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src/random )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src )

FIND_PACKAGE( BLAS REQUIRED)
SET (TEST_LINK_LIBS ${BLAS_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${Boost_LIBRARIES} cuv${LIB_SUFFIX})

SET (CUDA_TEST_DEVICE "0" CACHE STRING "Which CUDA device should the tests be run on, if multiple devices are present.")
add_definitions(-DCUDA_TEST_DEVICE=${CUDA_TEST_DEVICE})

ADD_EXECUTABLE( test_tensor tensor.cpp )
TARGET_LINK_LIBRARIES( test_tensor ${TEST_LINK_LIBS})

ADD_EXECUTABLE( test_tensor_serialization tensor_serialization.cpp )
TARGET_LINK_LIBRARIES( test_tensor_serialization ${TEST_LINK_LIBS})

ADD_EXECUTABLE( test_basic basic.cpp )
TARGET_LINK_LIBRARIES( test_basic ${TEST_LINK_LIBS})

ADD_EXECUTABLE( test_convert convert.cpp )
TARGET_LINK_LIBRARIES( test_convert ${TEST_LINK_LIBS})

ADD_EXECUTABLE( test_tensor_op tensor_op.cpp )
TARGET_LINK_LIBRARIES( test_tensor_op ${TEST_LINK_LIBS})

ADD_EXECUTABLE( test_theano_ops lib_theano_ops.cpp )
TARGET_LINK_LIBRARIES( test_theano_ops ${TEST_LINK_LIBS})

ADD_EXECUTABLE( test_tensor_op_speed tensor_op_speed.cpp )
TARGET_LINK_LIBRARIES( test_tensor_op_speed ${TEST_LINK_LIBS})

ADD_EXECUTABLE( test_optimize optimize.cpp )
TARGET_LINK_LIBRARIES( test_optimize ${TEST_LINK_LIBS})

ADD_EXECUTABLE( test_mat_op matrix_op.cpp )
#TARGET_LINK_LIBRARIES( test_mat_op cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_matrix_ops cuv_random ${CUDA_CUBLAS_LIBRARIES} )
TARGET_LINK_LIBRARIES( test_mat_op ${TEST_LINK_LIBS})

#ADD_EXECUTABLE( test_dia_mat dia_mat.cpp )
#TARGET_LINK_LIBRARIES( test_dia_mat ${TEST_LINK_LIBS})

#ADD_EXECUTABLE( test_spmv spmv.cpp )
#TARGET_LINK_LIBRARIES( test_spmv   ${TEST_LINK_LIBS})

#ADD_EXECUTABLE( test_spmv_speed spmv_speed.cpp )
#TARGET_LINK_LIBRARIES( test_spmv_speed   ${TEST_LINK_LIBS})

#ADD_EXECUTABLE( test_densedense_to_dia densedense_to_dia.cpp )
#TARGET_LINK_LIBRARIES( test_densedense_to_dia   ${TEST_LINK_LIBS})

#ADD_EXECUTABLE( test_densedense_to_dia_speed densedense_to_dia_speed.cpp )
#TARGET_LINK_LIBRARIES( test_densedense_to_dia_speed     ${TEST_LINK_LIBS})

ADD_EXECUTABLE( test_mat_op_speed matrix_op_speed.cpp )
TARGET_LINK_LIBRARIES( test_mat_op_speed ${TEST_LINK_LIBS})

ADD_EXECUTABLE( test_conv_op conv_op.cpp )
TARGET_LINK_LIBRARIES( test_conv_op ${TEST_LINK_LIBS})

ADD_EXECUTABLE( test_conv_op_speed conv_op_speed.cpp )
TARGET_LINK_LIBRARIES( test_conv_op_speed ${TEST_LINK_LIBS})

ADD_EXECUTABLE( test_random random.cpp )
TARGET_LINK_LIBRARIES( test_random ${TEST_LINK_LIBS})

ADD_EXECUTABLE( test_random_speed random_speed.cpp )
TARGET_LINK_LIBRARIES( test_random_speed ${TEST_LINK_LIBS})

ADD_EXECUTABLE( test_memory memory.cpp )
TARGET_LINK_LIBRARIES( test_memory ${TEST_LINK_LIBS})

ADD_EXECUTABLE( test_lib_rbm lib_rbm.cpp )
TARGET_LINK_LIBRARIES( test_lib_rbm ${TEST_LINK_LIBS})

ADD_EXECUTABLE( test_lib_kmeans lib_kmeans.cpp )
#TARGET_LINK_LIBRARIES( test_lib_kmeans cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_matrix_ops cuv_kmeans ${CUDA_CUBLAS_LIBRARIES} )
TARGET_LINK_LIBRARIES( test_lib_kmeans ${TEST_LINK_LIBS})

IF(CUV_CIMG_BINDINGS)
	FIND_PACKAGE( PNG REQUIRED)
	ADD_EXECUTABLE( test_lib_cimg cimg.cpp )
    SET(TEST_LINK_LIBS ${TEST_LINK_LIBS} X11 pthread ${PNG_LIBRARIES})
    TARGET_LINK_LIBRARIES( test_lib_cimg ${TEST_LINK_LIBS})

	ADD_EXECUTABLE( test_lib_sep_conv lib_sep_conv.cpp )
    TARGET_LINK_LIBRARIES( test_lib_sep_conv ${TEST_LINK_LIBS})

	ADD_EXECUTABLE( test_lib_intimg lib_intimg.cpp )
    TARGET_LINK_LIBRARIES( test_lib_intimg ${TEST_LINK_LIBS})

	ADD_EXECUTABLE( test_nlmeans lib_nlmean.cpp )
    TARGET_LINK_LIBRARIES( test_nlmeans ${TEST_LINK_LIBS})

	ADD_EXECUTABLE( test_hog lib_hog.cpp )
    TARGET_LINK_LIBRARIES( test_hog ${TEST_LINK_LIBS})

        FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

ENDIF(CUV_CIMG_BINDINGS)


if (PYTHONLIBS_FOUND )
TARGET_LINK_LIBRARIES(test_theano_ops ${CUV_LIBRARIES})
endif(PYTHONLIBS_FOUND )

ADD_TEST( tensor_serialization  "${CMAKE_BINARY_DIR}/src/tests/test_tensor_serialization" )
ADD_TEST( basics  "${CMAKE_BINARY_DIR}/src/tests/test_basic" )
ADD_TEST( convert "${CMAKE_BINARY_DIR}/src/tests/test_convert" )
ADD_TEST( tensor_op "${CMAKE_BINARY_DIR}/src/tests/test_tensor_op" )
ADD_TEST( optimize "${CMAKE_BINARY_DIR}/src/tests/test_optimize" )
ADD_TEST( matrix_op "${CMAKE_BINARY_DIR}/src/tests/test_mat_op" )
#ADD_TEST( dia_mat "${CMAKE_BINARY_DIR}/src/tests/test_dia_mat" )
#ADD_TEST( spmv "${CMAKE_BINARY_DIR}/src/tests/test_spmv" )
#ADD_TEST( densedense_to_dia "${CMAKE_BINARY_DIR}/src/tests/test_densedense_to_dia" )
ADD_TEST( random "${CMAKE_BINARY_DIR}/src/tests/test_random" )
ADD_TEST( lib_rbm "${CMAKE_BINARY_DIR}/src/tests/test_lib_rbm" )
ADD_TEST( lib_kmeans "${CMAKE_BINARY_DIR}/src/tests/test_lib_kmeans" )
ADD_TEST( conv_op "${CMAKE_BINARY_DIR}/src/tests/test_conv_op" )
ADD_TEST( theano_ops "${CMAKE_BINARY_DIR}/src/tests/test_theano_ops" )

# test for memory leak. takes to long and bug was fixed
#ADD_TEST( memory  "${CMAKE_BINARY_DIR}/src/tests/test_memory" )

# speed tests just take too long.
#ADD_TEST( tensor_op_speed "${CMAKE_BINARY_DIR}/tests/test_tensor_op_speed" )
#ADD_TEST( mat_op_speed "${CMAKE_BINARY_DIR}/tests/test_mat_op_speed" )
#ADD_TEST( spmv_speed "${CMAKE_BINARY_DIR}/tests/test_spmv_speed" )
#ADD_TEST( densedense_to_dia_speed "${CMAKE_BINARY_DIR}/tests/test_densedense_to_dia_speed" )
#ADD_TEST( conv_ops_speed "${CMAKE_BINARY_DIR}/tests/test_conv_op_speed" )
#ADD_TEST( random_speed "${CMAKE_BINARY_DIR}/tests/test_random_speed" )

IF(CUV_PYTHON_BINDINGS)
	SET(ENV{PYTHONPATH} ${CMAKE_BINARY_DIR}/python_bindings )
	ADD_TEST( load_py sh -c "PYTHONPATH=${CMAKE_BINARY_DIR}/src python -c 'import cuv_python as cp'" )

	FIND_PROGRAM(NOSETEST_EXECUTABLE nosetests)
	IF(NOSETEST_EXECUTABLE)
            ADD_TEST( nose_tests sh -c "PYTHONPATH=${CMAKE_BINARY_DIR}/src cd ${CMAKE_SOURCE_DIR}/src/tests/python ; nosetests" )
	ELSE(NOSETEST_EXECUTABLE)
		MESSAGE(STATUS "Warning: Did not find the nosetests binary, cannot run python tests")
	ENDIF(NOSETEST_EXECUTABLE)
ENDIF(CUV_PYTHON_BINDINGS)


