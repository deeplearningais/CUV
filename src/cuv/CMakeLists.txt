add_subdirectory(tools)
add_subdirectory(basics)
add_subdirectory(tensor_ops)
add_subdirectory(matrix_ops)
add_subdirectory(image_ops)
add_subdirectory(convolution_ops)
add_subdirectory(convert)
add_subdirectory(random)
add_subdirectory(libs)


CUDA_ADD_LIBRARY(cuv SHARED cuv.cpp)

TARGET_LINK_LIBRARIES(cuv 
	cuv_tools 
	cuv_basics 
	cuv_tensor_ops 
	cuv_matrix_ops 
	cuv_image_ops
	cuv_convolution_ops
	cuv_convert
	cuv_random
	cuv_rbm
	cuv_kmeans
	cuv_kernels
	cuv_intimg
	cuv_separable_conv
    tp_cudaconv2
	)
IF(CUV_CIMG_BINDINGS)
	TARGET_LINK_LIBRARIES(cuv cuv_cimg pthread X11 png)
ENDIF(CUV_CIMG_BINDINGS)
IF(CUV_SIFTGPU_BINDINGS)
	TARGET_LINK_LIBRARIES(cuv cuv_siftgpu)
ENDIF(CUV_SIFTGPU_BINDINGS)

install(TARGETS cuv
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static )
