ADD_DEFINITIONS(-DTHEANO_KERN_WID=0)



find_package (PythonLibs 2.7 REQUIRED)
if (PYTHONLIBS_FOUND )
    include_directories(${PYTHON_INCLUDE_DIRS} )

    CUDA_ADD_LIBRARY("tp_theano${LIB_SUFFIX}" SHARED
        convolutions.cu
        cuda_ndarray.cu
        )
    set_target_properties( "tp_theano${LIB_SUFFIX}" PROPERTIES VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR} SOVERSION 0 )


    TARGET_LINK_LIBRARIES("tp_theano${LIB_SUFFIX}" ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})

    install(TARGETS "tp_theano${LIB_SUFFIX}"
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)
endif(PYTHONLIBS_FOUND )
