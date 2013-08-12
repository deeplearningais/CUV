
find_package (PythonLibs 2.7 REQUIRED)
if (PYTHONLIBS_FOUND )
        include_directories(${PYTHON_INCLUDE_DIRS} )
    endif(PYTHONLIBS_FOUND )
