CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/../../include/cudaconv2 )
CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/../../include/nvmatrix )
CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/../../include/common )

CUDA_ADD_LIBRARY(tp_cudaconv2 SHARED
	conv_util.cu
	filter_acts.cu
	img_acts.cu
	weight_acts.cu
  )

TARGET_LINK_LIBRARIES(tp_cudaconv2 ${CUDA_LIBRARIES})

install(TARGETS tp_cudaconv2
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib/static)
