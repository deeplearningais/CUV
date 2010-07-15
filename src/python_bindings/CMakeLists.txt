set(Boost_USE_STATIC_LIBS FALSE)
FIND_PACKAGE( Boost 1.37 COMPONENTS date_time python REQUIRED )
FIND_PACKAGE( PythonLibs )
include_directories( 
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/basics
        ${CMAKE_SOURCE_DIR}/src/convert
        ${CMAKE_SOURCE_DIR}/src/vector_ops
        ${CMAKE_SOURCE_DIR}/src/matrix_ops
        ${CMAKE_SOURCE_DIR}/src/convolution_ops
        ${CMAKE_SOURCE_DIR}/src/random
        ${PYTHON_INCLUDE_PATH} 
        /usr/local/lib/python2.6/dist-packages/PyUblas-0.93.1-py2.6-linux-x86_64.egg/include
        )
ADD_LIBRARY(cuv_python SHARED 
	python_bindings.cpp
	export_vector.cpp
	export_dense_matrix.cpp
	export_dia_mat.cpp
	export_matrix_ops.cpp
	export_vector_ops.cpp
	export_random.cpp
	export_convolution_ops.cpp
	export_image_ops.cpp
	export_tools.cpp
    )
SET_TARGET_PROPERTIES(cuv_python PROPERTIES OUTPUT_NAME "cuv_python")
SET_TARGET_PROPERTIES(cuv_python PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES( cuv_python 
		cuv_tools
        cuv_basics
		cuv_vector_ops
		cuv_matrix_ops
		cuv_convert
		cuv_random
		cuv_matrix_ops
		cuv_convolution_ops
		cuv_image_ops
        #${Boost_LIBRARIES}
		${Boost_DATE_TIME_LIBRARY_RELEASE}
		${Boost_PYTHON_LIBRARY_RELEASE}
        ${PYTHON_LIBRARY}
        ${CUDA_CUBLAS_LIBRARIES}
)
install(TARGETS cuv_python
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib/static)

