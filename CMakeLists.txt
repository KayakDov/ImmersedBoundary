cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0135 NEW)

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
# Dependencies
# =========================
# Find CUDA toolkit (portable)
find_package(CUDAToolkit REQUIRED)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
# For Windows compatibility
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

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
        "${CMAKE_CURRENT_SOURCE_DIR}/shroud_project/wrapEigenBCGSolver_ImEq.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shroud_project/wrapEigenBCGSolver_eigen.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shroud_project/utilEigenBCGSolver.cpp"
)

set(SHROUD_F90
        "${CMAKE_CURRENT_SOURCE_DIR}/shroud_project/wrapfEigenBCGSolver_eigen.f90"
        "${CMAKE_CURRENT_SOURCE_DIR}/shroud_project/wrapfEigenBCGSolver_ImEq.f90"
        "${CMAKE_CURRENT_SOURCE_DIR}/shroud_project/wrapfeigenbcgsolver.f90"
)

# =========================
# Library Target
# =========================
add_library(CudaBandedLib STATIC
        ${ALL_SRC}
        ${SHROUD_CPP}
        ${SHROUD_F90}
        src/immersedBoundary/IBDataFactory.hpp
        src/solvers/EigenDecomp/EigenDecompForFortran.cpp
        src/solvers/EigenDecomp/EigenDecompForFortran.h
)

set_target_properties(CudaBandedLib PROPERTIES
        Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules
        POSITION_INDEPENDENT_CODE ON
)

target_include_directories(CudaBandedLib
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/deviceArrays/headers
        ${CMAKE_CURRENT_SOURCE_DIR}/src/deviceArrays/headers/sparse
        ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers
        ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/EigenDecomp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/immersedBoundary
        ${CMAKE_CURRENT_SOURCE_DIR}/src/poisson
        ${CMAKE_CURRENT_SOURCE_DIR}/shroud_project
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(CudaBandedLib
        PUBLIC
        CUDA::cublas
        CUDA::cusolver
        CUDA::cusparse
        CUDA::cuda_driver
)

# =========================
# Unit Test Executable
# =========================
enable_testing()

# We use the source lists directly to ensure GTest macros are compiled into the binary
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/Tests/*.cpp")

add_executable(run_unit_tests ${TEST_SOURCES})

set_target_properties(run_unit_tests PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Shared
)

# Link the core library and GoogleTest to the executable
target_link_libraries(run_unit_tests
        PRIVATE
        CudaBandedLib
        GTest::gtest
        GTest::gtest_main
)

# Discovery
include(GoogleTest)
gtest_discover_tests(run_unit_tests)

# =========================
# Install the library and Fortran modules for exFternal projects
# =========================

# Set a standard install directory for libraries and modules
set(INSTALL_LIB_DIR lib)
set(INSTALL_MOD_DIR include/fortran_modules)

# Install the static library
install(TARGETS CudaBandedLib
        ARCHIVE DESTINATION ${INSTALL_LIB_DIR}       # for STATIC libraries
        LIBRARY DESTINATION ${INSTALL_LIB_DIR}       # for shared, if ever
        RUNTIME DESTINATION bin                      # executables
)

# Install Fortran modules
# CMAKE_Fortran_MODULE_DIRECTORY must match where the .mod files were generated
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules/
        DESTINATION ${INSTALL_MOD_DIR}
        FILES_MATCHING PATTERN "*.mod"
)