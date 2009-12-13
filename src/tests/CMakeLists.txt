INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/basics )
INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/convert )

ADD_EXECUTABLE( test_basic basic.cpp )
TARGET_LINK_LIBRARIES( test_basic cuv_basics cuv_tools )

ADD_EXECUTABLE( test_convert convert.cpp )
TARGET_LINK_LIBRARIES( test_convert cuv_basics cuv_tools cuv_convert)


ADD_TEST( basics  "${CMAKE_BINARY_DIR}/tests/test_basic" )
ADD_TEST( convert "${CMAKE_BINARY_DIR}/tests/test_convert" )

