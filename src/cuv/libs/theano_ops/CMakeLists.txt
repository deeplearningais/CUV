
find_package (PythonLibs REQUIRED)
if (PYTHONLIBS_FOUND )
        include_directories(${PYTHON_INCLUDE_DIRS} )
    endif(PYTHONLIBS_FOUND )
