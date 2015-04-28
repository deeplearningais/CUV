ADD_DEFINITIONS(-DTHEANO_KERN_WID=0)
FIND_PACKAGE( BLAS REQUIRED)

add_subdirectory(tools)
add_subdirectory(basics)
add_subdirectory(tensor_ops)
add_subdirectory(matrix_ops)
add_subdirectory(image_ops)
add_subdirectory(convolution_ops)
add_subdirectory(convert)
add_subdirectory(random)
add_subdirectory(libs)

set(CUV_SRC 
    cuv.cpp
    matrix_ops/matrix_ops_reduce.cu
    #matrix_ops/diagonals.cu
    #matrix_ops/densedense_to_sparse.cu
    #matrix_ops/spmv.cu
    matrix_ops/matrix_ops.cu
    random/random.cu
    image_ops/move.cu
    image_ops/image_pyramid.cu
    tensor_ops/rprop.cu
    libs/hog/hog.cu
    libs/kernels/kernels.cu
    libs/separable_conv/separable_convolution.cu
    libs/opt/opt.cu
    libs/integral_image/integral_image.cu
    libs/nlmeans/conv3d.cu
    libs/nlmeans/nlmeans.cu
    libs/rbm/rbm.cu
    libs/kmeans/kmeans.cu
    libs/caffe/caffe.cu
    convert/convert.cu
    basics/cuda_array.cu
    basics/reference.cu
    basics/allocators.cu
    basics/memory.cu
    basics/io.cpp
    convolution_ops/convolution_ops.cu
    tools/progressbar.cpp
    tools/device_tools.cpp
    tools/cuPrintf.cu
    tools/cuv_general.cu
    ${TENSOR_OPS_INST}
    )


IF(CUV_CIMG_BINDINGS)
    SET(CUV_SRC ${CUV_SRC} libs/cimg/cuv_cimg.cu)
ENDIF(CUV_CIMG_BINDINGS)

find_package (PythonLibs 2.7 REQUIRED)
if (PYTHONLIBS_FOUND )
    #SET(CUV_SRC ${CUV_SRC} libs/theano_ops/theano_ops.cu)
    include_directories(${PYTHON_INCLUDE_DIRS} )
endif(PYTHONLIBS_FOUND )

CUDA_ADD_LIBRARY("cuv${LIB_SUFFIX}" SHARED ${CUV_SRC})
CUDA_ADD_CUBLAS_TO_TARGET(cuv${LIB_SUFFIX})

set_target_properties( "cuv${LIB_SUFFIX}" PROPERTIES VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR} SOVERSION 0 )


SET(CUV_LIBRARIES  tp_cudaconv2${LIB_SUFFIX} ${CUDA_LIBRARIES} ${CUDA_CUT_LIBRARY} ${BLAS_LIBRARIES})

if (PYTHONLIBS_FOUND )
    #SET(CUV_LIBRARIES  ${CUV_LIBRARIES} tp_theano${LIB_SUFFIX} ${PYTHON_LIBRARIES})
    SET(CUV_LIBRARIES  ${CUV_LIBRARIES} ${PYTHON_LIBRARIES})
endif(PYTHONLIBS_FOUND )


TARGET_LINK_LIBRARIES("cuv${LIB_SUFFIX}" ${CUV_LIBRARIES})

IF(CUV_CIMG_BINDINGS)
    TARGET_LINK_LIBRARIES("cuv${LIB_SUFFIX}" pthread X11 png)
ENDIF(CUV_CIMG_BINDINGS)

IF(CUV_SIFTGPU_BINDINGS)
    TARGET_LINK_LIBRARIES("cuv${LIB_SUFFIX}" cuv_siftgpu)
ENDIF(CUV_SIFTGPU_BINDINGS)

install(TARGETS "cuv${LIB_SUFFIX}"
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static )
