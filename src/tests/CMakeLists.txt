INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src/basics )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src/convert )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src/tensor_ops )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src/matrix_ops )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src/convolution_ops )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src/random )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src )

FIND_PACKAGE( BLAS REQUIRED)

SET (CUDA_TEST_DEVICE "0" CACHE STRING "Which CUDA device should the tests be run on, if multiple devices are present.")
add_definitions(-DCUDA_TEST_DEVICE=${CUDA_TEST_DEVICE})

ADD_EXECUTABLE( test_tensor tensor.cpp )
TARGET_LINK_LIBRARIES( test_tensor cuv_basics cuv_tools cuv_matrix_ops cuv_convert)

ADD_EXECUTABLE( test_basic basic.cpp )
TARGET_LINK_LIBRARIES( test_basic cuv_basics cuv_tools cuv_matrix_ops cuv_convert)

ADD_EXECUTABLE( test_convert convert.cpp )
TARGET_LINK_LIBRARIES( test_convert cuv_basics cuv_tools cuv_convert)

ADD_EXECUTABLE( test_tensor_op tensor_op.cpp )
TARGET_LINK_LIBRARIES( test_tensor_op cuv_basics cuv_tools cuv_convert cuv_tensor_ops)

ADD_EXECUTABLE( test_tensor_op_speed tensor_op_speed.cpp )
TARGET_LINK_LIBRARIES( test_tensor_op_speed cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_random )

ADD_EXECUTABLE( test_mat_op matrix_op.cpp )
TARGET_LINK_LIBRARIES( test_mat_op cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_matrix_ops cuv_random ${CUDA_CUBLAS_LIBRARIES} )

ADD_EXECUTABLE( test_dia_mat dia_mat.cpp )
TARGET_LINK_LIBRARIES( test_dia_mat cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_matrix_ops ${CUDA_CUBLAS_LIBRARIES} )

ADD_EXECUTABLE( test_spmv spmv.cpp )
TARGET_LINK_LIBRARIES( test_spmv         cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_matrix_ops ${CUDA_CUBLAS_LIBRARIES} )

ADD_EXECUTABLE( test_spmv_speed spmv_speed.cpp )
TARGET_LINK_LIBRARIES( test_spmv_speed   cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_matrix_ops ${CUDA_CUBLAS_LIBRARIES} )

ADD_EXECUTABLE( test_densedense_to_dia densedense_to_dia.cpp )
TARGET_LINK_LIBRARIES( test_densedense_to_dia     cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_matrix_ops ${CUDA_CUBLAS_LIBRARIES} )

ADD_EXECUTABLE( test_densedense_to_dia_speed densedense_to_dia_speed.cpp )
TARGET_LINK_LIBRARIES( test_densedense_to_dia_speed     cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_matrix_ops ${CUDA_CUBLAS_LIBRARIES} )

ADD_EXECUTABLE( test_mat_op_speed matrix_op_speed.cpp )
TARGET_LINK_LIBRARIES( test_mat_op_speed cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_matrix_ops cuv_random ${CUDA_CUBLAS_LIBRARIES} )

ADD_EXECUTABLE( test_conv_op conv_op.cpp )
TARGET_LINK_LIBRARIES( test_conv_op cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_matrix_ops cuv_convolution_ops cuv_random ${CUDA_CUBLAS_LIBRARIES} )

ADD_EXECUTABLE( test_conv_op_speed conv_op_speed.cpp )
TARGET_LINK_LIBRARIES( test_conv_op_speed cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_matrix_ops cuv_convolution_ops cuv_random ${CUDA_CUBLAS_LIBRARIES} ${BLAS_LIBRARIES} )

ADD_EXECUTABLE( test_random random.cpp )
TARGET_LINK_LIBRARIES( test_random cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_random)

ADD_EXECUTABLE( test_random_speed random_speed.cpp )
TARGET_LINK_LIBRARIES( test_random_speed cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_random)

ADD_EXECUTABLE( test_memory memory.cpp )
TARGET_LINK_LIBRARIES( test_memory cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_matrix_ops ${CUDA_CUBLAS_LIBRARIES} )

ADD_EXECUTABLE( test_lib_rbm lib_rbm.cpp )
TARGET_LINK_LIBRARIES( test_lib_rbm cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_matrix_ops cuv_rbm ${CUDA_CUBLAS_LIBRARIES} )


IF(CUV_CIMG_BINDINGS)
	FIND_PACKAGE( PNG REQUIRED)
	ADD_EXECUTABLE( test_lib_cimg cimg.cpp )
	TARGET_LINK_LIBRARIES( test_lib_cimg cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_matrix_ops cuv_cimg X11 pthread ${PNG_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} )

	ADD_EXECUTABLE( test_lib_sep_conv lib_sep_conv.cpp )
	TARGET_LINK_LIBRARIES( test_lib_sep_conv cuv_basics cuv_tools cuv_convert cuv_tensor_ops cuv_matrix_ops cuv_separable_conv cuv_cimg X11 pthread ${PNG_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} )
ENDIF(CUV_CIMG_BINDINGS)

ADD_TEST( basics  "${CMAKE_BINARY_DIR}/src/tests/test_basic" )
ADD_TEST( convert "${CMAKE_BINARY_DIR}/src/tests/test_convert" )
ADD_TEST( tensor_op "${CMAKE_BINARY_DIR}/src/tests/test_tensor_op" )
ADD_TEST( matrix_op "${CMAKE_BINARY_DIR}/src/tests/test_mat_op" )
ADD_TEST( dia_mat "${CMAKE_BINARY_DIR}/src/tests/test_dia_mat" )
ADD_TEST( spmv "${CMAKE_BINARY_DIR}/src/tests/test_spmv" )
ADD_TEST( densedense_to_dia "${CMAKE_BINARY_DIR}/src/tests/test_densedense_to_dia" )
ADD_TEST( random "${CMAKE_BINARY_DIR}/src/tests/test_random" )
ADD_TEST( lib_rbm "${CMAKE_BINARY_DIR}/src/tests/test_lib_rbm" )
ADD_TEST( conv_op "${CMAKE_BINARY_DIR}/src/tests/test_conv_op" )

# test for memory leak. takes to long and bug was fixed
#ADD_TEST( memory  "${CMAKE_BINARY_DIR}/src/tests/test_memory" )

# speed tests just take too long.
#ADD_TEST( tensor_op_speed "${CMAKE_BINARY_DIR}/tests/test_tensor_op_speed" )
#ADD_TEST( mat_op_speed "${CMAKE_BINARY_DIR}/tests/test_mat_op_speed" )
#ADD_TEST( spmv_speed "${CMAKE_BINARY_DIR}/tests/test_spmv_speed" )
#ADD_TEST( densedense_to_dia_speed "${CMAKE_BINARY_DIR}/tests/test_densedense_to_dia_speed" )
#ADD_TEST( conv_ops_speed "${CMAKE_BINARY_DIR}/tests/test_conv_op_speed" )
#ADD_TEST( random_speed "${CMAKE_BINARY_DIR}/tests/test_random_speed" )

SET(ENV{PYTHONPATH} ${CMAKE_BINARY_DIR}/python_bindings )
ADD_TEST( load_py sh -c "PYTHONPATH=${CMAKE_BINARY_DIR}/src python -c 'import cuv_python as cp'" )

