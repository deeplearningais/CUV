macro(unit_test NAME CXX_FILE FILES)
	set(PATH_FILES "")
	foreach(part ${FILES})
		set(PATH_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${part}" ${PATH_FILES})
	endforeach(part ${FILES})
	set(CXX_FILE_REAL "${CMAKE_CURRENT_SOURCE_DIR}/${CXX_FILE}")
	add_custom_command(
		OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.cxx"
		COMMAND ${CXXTEST_TESTGEN_EXECUTABLE} --error-printer -o "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.cxx" ${CXX_FILE_REAL}
		DEPENDS "${FILE}"
		)
	set(CXXTEST_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.cxx")
	add_executable("${NAME}" "${CXXTEST_OUTPUT}" ${PATH_FILES})
	target_link_libraries("${NAME}" ${CXXTEST_LINK_LIBS})
	add_test("${NAME}" "${EXECUTABLE_OUTPUT_PATH}/${NAME}")
endmacro(unit_test)


find_path(CXXTEST_INCLUDE_DIR cxxtest/TestSuite.h PATHS ${CMAKE_SOURCE_DIR}/third_party/cxxtest )
find_program(CXXTEST_TESTGEN_EXECUTABLE cxxtestgen.py
    PATHS ${CXXTEST_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CxxTest DEFAULT_MSG CXXTEST_INCLUDE_DIR)

set(CXXTEST_INCLUDE_DIRS ${CXXTEST_INCLUDE_DIR})

