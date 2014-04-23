# try to find the pyublas library

execute_process ( COMMAND python -c "import pyublas; print(pyublas.__path__[0])" OUTPUT_VARIABLE PYUBLAS_OUT OUTPUT_STRIP_TRAILING_WHITESPACE)

find_path(PYUBLAS_INCLUDE_DIR pyublas/numpy.hpp
          HINTS ${PYUBLAS_OUT}/../include ${PYUBLAS_OUT}/include /usr/local/include /usr/include)

#mark_as_advanced(PYUBLAS_INCLUDE_DIR)
