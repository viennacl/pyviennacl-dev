# Project setup
###############

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project(PyViennaCL)

set(VERSION_MAJOR 0)
set(VERSION_MINOR 9999)
set(VERSION_PATCH 99)
set(VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

SET(CMAKE_VERBOSE_MAKEFILE ON)   #uncomment this line to increase CMake verbosity

#SET(BOOSTPATH "C:/Program\ Files\ (x86)/boost/boost_1_42")    #Boost path for other systems - customize as needed

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Add visibility of headers
# Necessary for Qt-Creator usage.
################
file( GLOB_RECURSE MAKE_HEADERS_VISIBLE_SRC
    *.hpp *.h)
add_custom_target( MAKE_HEADERS_VISIBLE SOURCES ${MAKE_HEADERS_VISIBLE_SRC} )

# Subdirectories
################

add_subdirectory(external/boost_numpy)
add_subdirectory(tests)

# Common options
################

set(VIENNACL_SRC_DIST TRUE)
include(ViennaCLCommon)

# Set release build:
IF(DEFINED CMAKE_BUILD_TYPE)
 SET (CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
ELSE()
 SET (CMAKE_BUILD_TYPE Release)
ENDIF()

# Find prerequisites
####################

if(NOT DEFINED PYTHON_EXECUTABLE)
  set(
    PYTHON_EXECUTABLE "python"
    CACHE STRING
    "Change this to the path for your interpreter; for instance 'python3'")
endif()
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" ${CMAKE_SOURCE_DIR}/pyversions.py python
  OUTPUT_VARIABLE PYVERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" ${CMAKE_SOURCE_DIR}/pyversions.py boost
  OUTPUT_VARIABLE BOOST_COMPONENTS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
#set(Python_ADDITIONAL_VERSIONS ${PYVERSION})
find_package(PythonInterp ${PYVERSION} EXACT)
message(${PYTHON_VERSION_STRING})
if(DEFINED PYTHON_INCLUDE_DIR AND NOT DEFINED PYTHON_INCLUDE_DIR2)
  message("${PYTHON_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS}")
  set(PYTHON_INCLUDE_DIR2 ${PYTHON_INCLUDE_DIR})
endif()
find_package(PythonLibs ${PYTHON_VERSION_STRING} REQUIRED)
find_package(Numpy REQUIRED)

#SET(BOOST_INCLUDEDIR ${PROJECT_SOURCE_DIR}/external/boost_numpy)
#SET(BOOST_LIBRARYDIR ${PROJECT_BINARY_DIR}/lib)
SET(Boost_USE_STATIC_LIBS OFF)
if (BOOST_COMPONENTS)
  find_package(Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS})
else ()
  find_package(Boost REQUIRED COMPONENTS python)
endif ()

# Compile options
#################

#include_directories(BEFORE ${PROJECT_SOURCE_DIR})

include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/external/boost_numpy
  ${PROJECT_SOURCE_DIR}/external/viennacl-dev
  ${NUMPY_INCLUDE_DIRS}
)

get_property(INCLUDE_DIRS DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY INCLUDE_DIRECTORIES)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  INCLUDE_DIRECTORIES("/opt/local/include")
  set(CMAKE_EXE_LINKER_FLAGS "-framework OpenCL")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}")

configure_file(${SETUP_PY_IN} ${SETUP_PY})

add_custom_command(
  OUTPUT ${OUTPUT}/build-stamp
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/src"
                                             "${CMAKE_CURRENT_BINARY_DIR}/src"
  COMMAND ${PYTHON_EXECUTABLE} setup.py build
  COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}/build-stamp
  DEPENDS ${DEPS} boost_numpy
)

add_custom_command(
  OUTPUT ${OUTPUT}/doc-stamp
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/doc"
                                             "${CMAKE_CURRENT_BINARY_DIR}/doc"
  COMMAND ${PYTHON_EXECUTABLE} doc/build-doc.py
  COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}/doc-stamp
  DEPENDS ${DEPS} build-stamp
)

add_custom_target(target ALL DEPENDS ${OUTPUT}/doc-stamp)

install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install --install-layout deb --prefix \$ENV{DESTDIR}/${CMAKE_INSTALL_PREFIX} )")
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory \"\$ENV{DESTDIR}/${CMAKE_INSTALL_PREFIX}/share/doc/pyviennacl-doc/\" )")
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory \"${CMAKE_CURRENT_BINARY_DIR}/build/doc\" \"\$ENV{DESTDIR}/${CMAKE_INSTALL_PREFIX}/share/doc/pyviennacl/\" )")

# Dist target
#############

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/dist")

set(DIST_SOURCES_DIR "${PROJECT_BINARY_DIR}/dist/pyviennacl-${VERSION}")

foreach(f LICENSE
    pyversions.py
    README.dist
    setup.py.in
    CMakeLists.txt
    cmake/FindNumpy.cmake # NumPy for PyViennaCL
    cmake/FindOpenCL.cmake
    cmake/ViennaCLCommon.cmake
    build/README)
  configure_file(${f} "${DIST_SOURCES_DIR}/${f}" COPYONLY)
endforeach()

add_custom_target(dist
  
  # Set up directory layout:
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${DIST_SOURCES_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${DIST_SOURCES_DIR}/external"
  
  #COMMAND "${CMAKE_COMMAND}" -E copy_directory
  #"${PROJECT_SOURCE_DIR}/debian" "${DIST_SOURCES_DIR}/debian"
  
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
     "${PROJECT_SOURCE_DIR}/doc" "${DIST_SOURCES_DIR}/doc"
  
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
     "${PROJECT_SOURCE_DIR}/external/boost_numpy"
     "${DIST_SOURCES_DIR}/external/boost_numpy"

  COMMAND "${CMAKE_COMMAND}" -E copy_directory
     "${PROJECT_SOURCE_DIR}/external/viennacl-dev/viennacl"
     "${DIST_SOURCES_DIR}/external/viennacl-dev/viennacl"
  
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
     "${PROJECT_SOURCE_DIR}/src" "${DIST_SOURCES_DIR}/src"

  COMMAND "${CMAKE_COMMAND}" -E copy_directory
     "${PROJECT_SOURCE_DIR}/tests" "${DIST_SOURCES_DIR}/tests"
    
  # Create archive:
  COMMAND tar cJf pyviennacl_${VERSION}.orig.tar.xz
  pyviennacl-${VERSION}
  WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/dist"
  COMMENT "Generating dist/pyviennacl_${VERSION}.orig.tar.xz"
  VERBATIM)

