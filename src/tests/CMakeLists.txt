INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/basics )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/convert )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/vector_ops )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/matrix_ops )

ADD_EXECUTABLE( test_basic basic.cpp )
TARGET_LINK_LIBRARIES( test_basic cuv_basics cuv_tools )

ADD_EXECUTABLE( test_convert convert.cpp )
TARGET_LINK_LIBRARIES( test_convert cuv_basics cuv_tools cuv_convert)

ADD_EXECUTABLE( test_vec_op vec_op.cpp )
TARGET_LINK_LIBRARIES( test_vec_op cuv_basics cuv_tools cuv_convert cuv_vector_ops)

ADD_EXECUTABLE( test_mat_op matrix_op.cpp )
TARGET_LINK_LIBRARIES( test_mat_op cuv_basics cuv_tools cuv_convert cuv_vector_ops)

ADD_EXECUTABLE( test_random random.cpp )
TARGET_LINK_LIBRARIES( test_random cuv_basics cuv_tools cuv_convert cuv_vector_ops cuv_random)



ADD_TEST( basics  "${CMAKE_BINARY_DIR}/tests/test_basic" )
ADD_TEST( convert "${CMAKE_BINARY_DIR}/tests/test_convert" )
ADD_TEST( vec_op "${CMAKE_BINARY_DIR}/tests/test_vec_op" )
ADD_TEST( random "${CMAKE_BINARY_DIR}/tests/test_random" )

SET(ENV{PYTHONPATH} ${CMAKE_BINARY_DIR}/python_bindings )
ADD_TEST( load_py sh -c "PYTHONPATH=${CMAKE_BINARY_DIR}/python_bindings python -c 'import cuv_python as cp'" )

