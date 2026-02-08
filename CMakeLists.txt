cmake_minimum_required(VERSION 3.18)

# 1. ARCHITECTURE: Set this BEFORE project() to ensure it applies to all tests
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES all-major) # Better than 'all' for faster compile times
endif()
# Project
project(CudaBanded LANGUAGES CXX CUDA Fortran)

# =========================
# Standards
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_Fortran_STANDARD 2008)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# =========================
# CUDA configuration
# =========================


# Find CUDA toolkit (portable)
find_package(CUDAToolkit REQUIRED)

# =========================
# Source collection
# =========================
file(GLOB_RECURSE ALL_SRC
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

list(FILTER ALL_SRC EXCLUDE REGEX ".*/main\\.cu$")

set(SHROUD_CPP
        shroud_project/wrapFortranBindings.cpp
        shroud_project/utilFortranBindings.cpp
        shroud_project/wrapffortranbindings.f90
        src/deviceArrays/headers/sparse/SparseMat.h
        src/deviceArrays/defFiles/SparseMat.cu
        src/deviceArrays/defFiles/SparseCSR.cpp
        src/deviceArrays/headers/sparse/SparseCSR.h
        src/deviceArrays/defFiles/SparseCOO.cpp
        src/deviceArrays/headers/sparse/SparseCOO.h
        src/solvers/Thomas.cu
        src/solvers/Thomas.cuh
)

# =========================
# Library
# =========================
add_library(CudaBandedLib STATIC
        ${ALL_SRC}
        ${SHROUD_CPP}
        ${FORTRAN_WRAPPER}
        src/immersedBoundary/IBDataFactory.hpp
)

set_target_properties(CudaBandedLib PROPERTIES
        Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules
)

target_include_directories(CudaBandedLib
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/deviceArrays/headers
        ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/BiCGSTAB
        ${CMAKE_CURRENT_SOURCE_DIR}/src/poisson
        ${CMAKE_CURRENT_SOURCE_DIR}/shroud_project
)

target_link_libraries(CudaBandedLib
        PUBLIC
        CUDA::cublas
        CUDA::cusolver
        CUDA::cusparse
        CUDA::cuda_driver
)

# =========================
# Executable
# =========================
add_executable(testMethods
        ${ALL_SRC}
        shroud_project/wrapFortranBindings.cpp
)

set_target_properties(testMethods PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(testMethods
        PRIVATE
        CudaBandedLib
)
