#if(NOT DEFINED PYTHON_EXECUTABLE)
#  set(
#    PYTHON_EXECUTABLE "python"
#    CACHE STRING
#    "Change this to the path for your interpreter; for instance 'python3'")
#endif()

#execute_process(
#  COMMAND "${PYTHON_EXECUTABLE}" ${CMAKE_SOURCE_DIR}/pyviennacl/pyversions.py python
#  OUTPUT_VARIABLE PYVERSION
#  OUTPUT_STRIP_TRAILING_WHITESPACE
#  )

#find_package(PythonInterp ${PYVERSION} EXACT)

add_test(NAME blas3_prod_py
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tests/pyviennacl"
  COMMAND sh -c "PYTHONPATH=${PYTHONPATH}:$ENV{PYTHONPATH} ${PYTHON_EXECUTABLE} blas3_prod.py")

add_test(NAME blas3_solve_py
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tests/pyviennacl"
  COMMAND sh -c "PYTHONPATH=${PYTHONPATH}:$ENV{PYTHONPATH} ${PYTHON_EXECUTABLE} blas3_solve.py")

# TODO -- GPU SCALAR NOT WORKING YET
#add_test(NAME scalar_operations_py
#  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tests/pyviennacl"
#  COMMAND sh -c "PYTHONPATH=${PYTHONPATH}:$ENV{PYTHONPATH} ${PYTHON_EXECUTABLE} scalar_operations.py")

add_test(NAME vector_operations_py
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tests/pyviennacl"
  COMMAND sh -c "PYTHONPATH=${PYTHONPATH}:$ENV{PYTHONPATH} ${PYTHON_EXECUTABLE} vector_operations.py")

add_test(NAME matrix_operations_py
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tests/pyviennacl"
  COMMAND sh -c "PYTHONPATH=${PYTHONPATH}:$ENV{PYTHONPATH} ${PYTHON_EXECUTABLE} matrix_operations.py")

