set(Boost_USE_STATIC_LIBS FALSE)
FIND_PACKAGE( Boost 1.37 COMPONENTS date_time python REQUIRED )
FIND_PACKAGE( PythonLibs )
FIND_PACKAGE( PythonInterp )
FIND_PACKAGE( BLAS REQUIRED)
if(PYTHON_VERSION_STRING)
    set(PYTHON_VERSION_MAJOR_MINOR "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
    string(REGEX MATCH "[0-9]+.[0-9]+.[0-9]+" PYTHON_VERSION_FULL "${PYTHON_VERSION_STRING}")
else()
    execute_process(COMMAND ${PYTHON_EXECUTABLE} --version
        ERROR_VARIABLE PYTHON_VERSION_FULL
        ERROR_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "[0-9]+.[0-9]+" PYTHON_VERSION_MAJOR_MINOR "${PYTHON_VERSION_FULL}")
    string(REGEX MATCH "[0-9]+.[0-9]+.[0-9]+" PYTHON_VERSION_FULL "${PYTHON_VERSION_FULL}")
endif()


FIND_PACKAGE( PyUblas REQUIRED )
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

    FIND_PACKAGE(PythonLibs REQUIRED)
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
        #export_libs_hog.cpp
    )

CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
#CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/.. COPYONLY)

SET_TARGET_PROPERTIES(cuv_python PROPERTIES OUTPUT_NAME "_cuv_python")
SET_TARGET_PROPERTIES(cuv_python PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES(cuv_python cuv
        #${Boost_LIBRARIES}
        ${Boost_DATE_TIME_LIBRARY_RELEASE}
        ${Boost_PYTHON_LIBRARY_RELEASE}
        ${PYTHON_LIBRARY}
        ${CUDA_CUBLAS_LIBRARIES}
        ${BLAS_LIBRARIES}
)

IF( CUV_CIMG_BINDINGS )
	FIND_PACKAGE( PNG REQUIRED)
	TARGET_LINK_LIBRARIES(cuv_python X11 pthread ${PNG_LIBRARIES})
ENDIF( CUV_CIMG_BINDINGS )

#install(TARGETS cuv_python
#        RUNTIME DESTINATION bin
#        LIBRARY DESTINATION lib
#        ARCHIVE DESTINATION lib/static)


execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import *; print get_python_lib()"
    RESULT_VARIABLE PYTHON_CVPY_PROCESS
    OUTPUT_VARIABLE PYTHON_STD_PACKAGES_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if("${PYTHON_STD_PACKAGES_PATH}" MATCHES "site-packages")
    set(_PYTHON_PACKAGES_PATH "python${PYTHON_VERSION_MAJOR_MINOR}/site-packages")
else() #debian based assumed, install to the dist-packages.
    set(_PYTHON_PACKAGES_PATH "python${PYTHON_VERSION_MAJOR_MINOR}/dist-packages")
endif()
if(EXISTS "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}/${PYTHON_PACKAGES_PATH}")
    set(_PYTHON_PACKAGES_PATH "lib${LIB_SUFFIX}/${_PYTHON_PACKAGES_PATH}")
else()
    set(_PYTHON_PACKAGES_PATH "lib/${_PYTHON_PACKAGES_PATH}")
endif()

SET(PYTHON_VERSION_MAJOR_MINOR ${PYTHON_VERSION_MAJOR_MINOR} CACHE STRING "Python version used")
SET(_PYTHON_PACKAGES_PATH ${_PYTHON_PACKAGES_PATH} CACHE PATH "Where to install python module to")

#execute_process ( COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
install(TARGETS cuv_python  DESTINATION ${_PYTHON_PACKAGES_PATH}/cuv_python${LIB_SUFFIX})
install(FILES   __init__.py DESTINATION ${_PYTHON_PACKAGES_PATH}/cuv_python${LIB_SUFFIX})

#INCLUDE(PythonMacros)
#GET_TARGET_PROPERTY(CUV_PYTHON_BIN cuv_python LOCATION)
#PYTHON_INSTALL_MODULE(cuv_python ${CUV_PYTHON_BIN} __init__.py)
