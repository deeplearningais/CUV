INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/basics )

ADD_EXECUTABLE( test_basic basic.cpp )
TARGET_LINK_LIBRARIES( test_basic cuv_basics cuv_tools )


ADD_TEST( basics "${CMAKE_BINARY_DIR}/tests/test_basic" )

