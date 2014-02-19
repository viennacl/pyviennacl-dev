include(CTest)
include(CMakeDependentOption)

# User options
##############

option(ENABLE_CUDA "Use the CUDA backend" OFF)

option(ENABLE_OPENCL "Use the OpenCL backend" ON)

option(ENABLE_OPENMP "Use OpenMP acceleration" OFF)

# Find prerequisites
####################

# Boost:
IF (BOOSTPATH)
 SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} ${BOOSTPATH})
 SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "${BOOSTPATH}/lib")
 SET(BOOST_ROOT ${BOOSTPATH})
ENDIF (BOOSTPATH)

if (ENABLE_CUDA)
   find_package(CUDA REQUIRED)
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_13 -DVIENNACL_WITH_CUDA)
endif(ENABLE_CUDA)

if (ENABLE_OPENCL)
   find_package(OpenCL REQUIRED)
endif(ENABLE_OPENCL)

if (ENABLE_OPENMP)
   find_package(OpenMP REQUIRED)
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -DVIENNACL_WITH_OPENMP")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DVIENNACL_WITH_OPENMP")
   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(ENABLE_OPENMP)

include_directories(
   ${PROJECT_SOURCE_DIR}
   ${OPENCL_INCLUDE_DIRS})

# Set high warning level on GCC
if(ENABLE_PEDANTIC_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
endif()

# Disable Warning 4996 (std::copy is unsafe ...) on Visual Studio
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996")
endif()


# Export
########

configure_file(cmake/FindOpenCL.cmake
   ${PROJECT_BINARY_DIR}/FindOpenCL.cmake COPYONLY)

if (CMAKE_MINOR_VERSION GREATER 6)  # export(PACKAGE ...) introduced with CMake 2.8.0
  export(PACKAGE PyViennaCL)
endif()

# Install
#########

#install(FILES
#   ${PROJECT_BINARY_DIR}/FindOpenCL.cmake
#   DESTINATION ${INSTALL_CMAKE_DIR} COMPONENT dev)
