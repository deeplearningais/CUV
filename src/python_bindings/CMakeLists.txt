set(Boost_USE_STATIC_LIBS FALSE)
FIND_PACKAGE( Boost 1.37 COMPONENTS date_time python REQUIRED )
FIND_PACKAGE( PythonLibs )
include_directories( 
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/basics
        ${CMAKE_SOURCE_DIR}/convert
        ${CMAKE_SOURCE_DIR}/vector_ops
        ${CMAKE_SOURCE_DIR}/random
        ${PYTHON_INCLUDE_PATH} 
        /usr/local/lib/python2.6/dist-packages/PyUblas-0.93.1-py2.6-linux-x86_64.egg/include
        )
ADD_LIBRARY(cuv_python SHARED 
	python_bindings.cpp
	export_vector.cpp
    )
SET_TARGET_PROPERTIES(cuv_python PROPERTIES OUTPUT_NAME "cuv_python")
SET_TARGET_PROPERTIES(cuv_python PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES( cuv_python 
		cuv_tools
        cuv_basics
		cuv_vector_ops
		cuv_convert
		cuv_random
        ${Boost_LIBRARIES}
        ${PYTHON_LIBRARY}
        ${CUDA_CUBLAS_LIBRARIES}
)

