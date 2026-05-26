cmake_minimum_required(VERSION 3.23)
cmake_policy(SET CMP0135 NEW)

if(WIN32)
    # 1. Apply /Oy- ONLY to C and C++ so NVCC ignores it
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/Oy->")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/Oy->")

    # 2. Bypass the strict MSVC STL vs. Intel Compiler version check
    add_compile_definitions(_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)
else()
    add_compile_options(-fno-omit-frame-pointer)
endif()

# 1. ARCHITECTURE: Set this BEFORE project() to ensure it applies to all tests
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES all-major) # Better than 'all' for faster compile times
endif()

# Project
project(CudaBanded LANGUAGES C CXX CUDA Fortran)

include(FortranCInterface)
FortranCInterface_VERIFY(CXX) # Fails build if they can't "talk" to each other

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
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.f90"
)

file(GLOB SHROUD_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/shroud_project/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shroud_project/*.f90"
)

list(FILTER LIB_SOURCES EXCLUDE REGEX "[/\\\\]main\\.cu$")

# =========================
# Library Target
# =========================
add_library(CudaBandedLib STATIC
        ${LIB_SOURCES}
        ${SHROUD_SOURCES}
)

set_target_properties(CudaBandedLib PROPERTIES
        Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Shared
)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules")

target_include_directories(CudaBandedLib
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/shroud_project
        "${CMAKE_CURRENT_BINARY_DIR}/modules"
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(CudaBandedLib
        INTERFACE
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

add_executable(run_unit_tests ${TEST_SOURCES}
        Tests/testPrimes.h)

set_target_properties(run_unit_tests PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared
)

target_link_libraries(run_unit_tests
        PRIVATE
        CudaBandedLib
        GTest::gtest
        GTest::gtest_main
)

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
if(NOT MSVC)
    target_link_options(run_unit_tests PRIVATE "-Wl,--no-undefined")
endif ()