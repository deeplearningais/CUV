set(Boost_USE_STATIC_LIBS FALSE)
FIND_PACKAGE( Boost 1.37 COMPONENTS date_time python REQUIRED )
FIND_PACKAGE( PythonLibs )
SET(PYUBLAS_INCLUDE_DIR "/usr/local/lib/python2.6/dist-packages/PyUblas-0.93.1-py2.6-linux-x86_64.egg/include" CACHE STRING "PyUblas include directory") 
include_directories( 
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/basics
        ${CMAKE_SOURCE_DIR}/src/convert
        ${CMAKE_SOURCE_DIR}/src/tensor_ops
        ${CMAKE_SOURCE_DIR}/src/matrix_ops
        ${CMAKE_SOURCE_DIR}/src/convolution_ops
        ${CMAKE_SOURCE_DIR}/src/random
        ${PYTHON_INCLUDE_PATH} 
        ${PYUBLAS_INCLUDE_DIR}
        )

FIND_PACKAGE(PythonLibs)
PYTHON_ADD_MODULE(cuv_python SHARED 
        python_bindings.cpp
        export_tensor.cpp
        export_cuda_array.cpp
        #export_dia_mat.cpp
        export_matrix_ops.cpp
        export_tensor_ops.cpp
        export_random.cpp
        export_convolution_ops.cpp
        export_image_ops.cpp
        export_tools.cpp
        export_libs_rbm.cpp
        export_libs_kmeans.cpp
        export_libs_kernels.cpp
        export_libs_cimg.cpp
        export_libs_hog.cpp
    )

CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

SET_TARGET_PROPERTIES(cuv_python PROPERTIES OUTPUT_NAME "_cuv_python")
SET_TARGET_PROPERTIES(cuv_python PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES(cuv_python cuv
        #${Boost_LIBRARIES}
        ${Boost_DATE_TIME_LIBRARY_RELEASE}
        ${Boost_PYTHON_LIBRARY_RELEASE}
        ${PYTHON_LIBRARY}
        ${CUDA_CUBLAS_LIBRARIES}
)

IF( CUV_CIMG_BINDINGS )
	FIND_PACKAGE( PNG REQUIRED)
	TARGET_LINK_LIBRARIES(cuv_python X11 pthread ${PNG_LIBRARIES})
ENDIF( CUV_CIMG_BINDINGS )

#install(TARGETS cuv_python
#        RUNTIME DESTINATION bin
#        LIBRARY DESTINATION lib
#        ARCHIVE DESTINATION lib/static)

execute_process ( COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
install(TARGETS cuv_python  DESTINATION ${PYTHON_SITE_PACKAGES}/cuv_python${LIB_SUFFIX})
install(FILES   __init__.py DESTINATION ${PYTHON_SITE_PACKAGES}/cuv_python${LIB_SUFFIX})

