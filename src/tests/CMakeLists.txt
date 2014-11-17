include (CMakeParseArguments)
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

SET(CUV_TESTS "")
SET(CUV_SPEED_TESTS "")
FUNCTION(cuv_add_test)
    set(options SPEEDTEST)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES)
    cmake_parse_arguments(CT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    #MESSAGE(STATUS "cuv_add_test: ${CT_NAME} <-- ${CT_SOURCES} (${TEST_LINK_LIBS})")

    ADD_EXECUTABLE(${CT_NAME} ${CT_SOURCES})
    TARGET_LINK_LIBRARIES( ${CT_NAME} ${TEST_LINK_LIBS})
    IF(CT_SPEEDTEST)
        # Speed tests tend to run forever, so they should not be included in
        # ctest runs.
        SET(CUV_SPEED_TESTS ${CUV_SPEED_TESTS} ${CT_NAME} PARENT_SCOPE)
    ELSE()
        SET(CUV_TESTS ${CUV_TESTS} ${CT_NAME} PARENT_SCOPE)
        ADD_TEST(${CT_NAME}  "${CMAKE_CURRENT_BINARY_DIR}/${CT_NAME}")
    ENDIF()
ENDFUNCTION(cuv_add_test)

cuv_add_test( NAME tensor SOURCES tensor.cpp)
cuv_add_test( NAME alloc SOURCES alloc.cpp )
cuv_add_test( NAME tensor_serialization SOURCES tensor_serialization.cpp )
cuv_add_test( NAME basic SOURCES basic.cpp )
cuv_add_test( NAME convert SOURCES convert.cpp )
cuv_add_test( NAME tensor_op SOURCES tensor_op.cpp )
cuv_add_test( NAME theano_ops SOURCES lib_theano_ops.cpp )
cuv_add_test( NAME tensor_op_speed SOURCES tensor_op_speed.cpp  SPEEDTEST )
cuv_add_test( NAME optimize SOURCES optimize.cpp )
cuv_add_test( NAME mat_op SOURCES matrix_op.cpp )

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

cuv_add_test( NAME mat_op_speed SOURCES matrix_op_speed.cpp  SPEEDTEST )
cuv_add_test( NAME conv_op SOURCES conv_op.cpp )
cuv_add_test( NAME conv_op_speed SOURCES conv_op_speed.cpp SPEEDTEST)
cuv_add_test( NAME random SOURCES random.cpp )
cuv_add_test( NAME random_speed SOURCES random_speed.cpp SPEEDTEST)
#cuv_add_test( NAME memory SOURCES memory.cpp )  # runs forever.
cuv_add_test( NAME lib_rbm SOURCES lib_rbm.cpp )
cuv_add_test( NAME lib_kmeans SOURCES lib_kmeans.cpp )

IF(CUV_CIMG_BINDINGS)
	FIND_PACKAGE( PNG REQUIRED)
    SET(TEST_LINK_LIBS ${TEST_LINK_LIBS} X11 pthread ${PNG_LIBRARIES})

    cuv_add_test( NAME lib_cimg SOURCES cimg.cpp )
    cuv_add_test( NAME lib_sep_conv SOURCES lib_sep_conv.cpp )
    cuv_add_test( NAME lib_intimg SOURCES lib_intimg.cpp )
    cuv_add_test( NAME nlmeans SOURCES lib_nlmean.cpp )
    cuv_add_test( NAME hog SOURCES lib_hog.cpp )
    FILE(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
ENDIF(CUV_CIMG_BINDINGS)


if (PYTHONLIBS_FOUND )
    TARGET_LINK_LIBRARIES(test_theano_ops ${CUV_LIBRARIES} ${PYTHON_LIBS})
endif(PYTHONLIBS_FOUND )

IF(CUV_PYTHON_BINDINGS)
	SET(ENV{PYTHONPATH} ${CMAKE_BINARY_DIR}/python_bindings )
	ADD_TEST( load_py sh -c "PYTHONPATH=${CMAKE_BINARY_DIR}/src ${PYTHON_EXECUTABLE} -c 'import cuv_python as cp'" )

	FIND_PROGRAM(NOSETEST_EXECUTABLE nosetests)
	IF(NOSETEST_EXECUTABLE)
            ADD_TEST( nose_tests sh -c "PYTHONPATH=${CMAKE_BINARY_DIR}/src cd ${CMAKE_SOURCE_DIR}/src/tests/python ; nosetests" )
	ELSE(NOSETEST_EXECUTABLE)
		MESSAGE(STATUS "Warning: Did not find the nosetests binary, cannot run python tests")
	ENDIF(NOSETEST_EXECUTABLE)
ENDIF(CUV_PYTHON_BINDINGS)

ADD_CUSTOM_TARGET(buildtests DEPENDS ${CUV_TESTS})
ADD_CUSTOM_TARGET(buildspeedtests DEPENDS ${CUV_SPEED_TESTS})
ADD_CUSTOM_TARGET(runtests 
    DEPENDS buildtests 
    POST_BUILD_COMMAND ctest ARGS --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

