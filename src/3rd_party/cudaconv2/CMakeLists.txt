CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include/nvmatrix )
CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include/common )
CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include/cudaconv2 )
CUDA_ADD_LIBRARY(tp_cudaconv2 SHARED
	include/common/matrix.cpp
	src/nvmatrix/nvmatrix.cu
	src/nvmatrix/nvmatrix_kernels.cu
	src/cudaconv2/conv_util.cu
	src/cudaconv2/filter_acts.cu
	src/cudaconv2/img_acts.cu
	src/cudaconv2/weight_acts.cu
  )

install(TARGETS tp_cudaconv2
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib/static)
