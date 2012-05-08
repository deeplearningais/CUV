FIND_PACKAGE( BLAS REQUIRED)

ADD_CUSTOM_COMMAND(
	OUTPUT  ${CMAKE_CURRENT_SOURCE_DIR}/spmv_dia_kernel_inst.cuh
	COMMAND perl ${CMAKE_CURRENT_SOURCE_DIR}/make_spmv_header.pl ${CMAKE_CURRENT_SOURCE_DIR}/spmv.cu ${CMAKE_CURRENT_SOURCE_DIR}/spmv_kernel.cuh ${CMAKE_CURRENT_BINARY_DIR}/spmv_dia_kernel_inst.cuh dia
	COMMAND cmake -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/spmv_dia_kernel_inst.cuh ${CMAKE_CURRENT_SOURCE_DIR}/spmv_dia_kernel_inst.cuh
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/spmv.cu ${CMAKE_CURRENT_SOURCE_DIR}/spmv_kernel.cuh ${CMAKE_CURRENT_SOURCE_DIR}/make_spmv_header.pl
	)

ADD_CUSTOM_TARGET(
	generateheader
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/spmv_dia_kernel_inst.cuh
	)

CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_BINARY_DIR} )
CUDA_INCLUDE_DIRECTORIES( ${CBLAS_INCLUDE_DIRECTORY} )
CUDA_INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} )
CUDA_INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src/basics )
CUDA_INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src/tensor_ops )
CUDA_ADD_LIBRARY(cuv_matrix_ops SHARED
	densedense_to_sparse.cu
	matrix_ops.cu
	matrix_ops_reduce.cu
	diagonals.cu
	spmv.cu
  )
ADD_DEPENDENCIES(cuv_matrix_ops generateheader)

CUDA_ADD_CUBLAS_TO_TARGET(cuv_matrix_ops)
TARGET_LINK_LIBRARIES(cuv_matrix_ops ${CUDA_LIBRARIES} ${CUDA_CUT_LIBRARY} ${BLAS_LIBRARIES} cuv_tensor_ops )
install(TARGETS cuv_matrix_ops
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib/static)


