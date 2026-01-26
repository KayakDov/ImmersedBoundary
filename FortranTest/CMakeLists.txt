cmake_minimum_required(VERSION 3.18)

# 1. Let CMake find gfortran/g++ automatically
project(FortranCudaProject LANGUAGES Fortran CXX)

# Where Fortran modules go
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod)

# Paths to Shroud wrapper + static C++ library
set(SHROUD_WRAPF "${CMAKE_SOURCE_DIR}/../shroud_project/wrapffortranbindings.f90")
set(CUDA_LIB "${CMAKE_SOURCE_DIR}/../cmake-build-debug/libCudaBandedLib.a")

# Find the standard CUDA Toolkit (nvcc)
find_package(CUDAToolkit REQUIRED)

# Build the Fortran executable
add_executable(fortran_app
        main.f90
        ${SHROUD_WRAPF}
)

# Standard Include directories
target_include_directories(fortran_app PRIVATE
        ${CMAKE_Fortran_MODULE_DIRECTORY}
        ${CUDAToolkit_INCLUDE_DIRS}
)

# Use gfortran as linker
set_target_properties(fortran_app PROPERTIES LINKER_LANGUAGE Fortran)

# Link C++ library and CUDA runtimes
target_link_libraries(fortran_app PRIVATE
        ${CUDA_LIB}
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        CUDA::cusparse
        stdc++               # CRITICAL: Links the C++ standard library
)