cmake_minimum_required(VERSION 3.23)
cmake_policy(SET CMP0135 NEW)

# =========================
# Hardware / Toolkit Alignment
# =========================
# 1. Point CMake to the root of your SDK
set(CUDA_SDK_ROOT "/opt/nvidia/hpc_sdk/Linux_x86_64/26.3/cuda")

# 2. Specify the exact folder version available in your SDK that satisfies your driver limit (<= 12.8)
set(TARGET_CUDA_VERSION "12.6")

# 3. Force CMake to use this specific toolkit path (FORCE ensures it overwrites old cache)
set(CUDAToolkit_ROOT "${CUDA_SDK_ROOT}/${TARGET_CUDA_VERSION}" CACHE PATH "Preferred CUDA version" FORCE)

# 4. ARCHITECTURE: Set this BEFORE project() to ensure it applies globally
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES all-major) # Better than 'all' for faster compile times
endif()

# =========================
# Project Initialization
# =========================
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(CudaBanded LANGUAGES C CXX CUDA Fortran)

if(WIN32)
    # Apply /Oy- ONLY to C and C++ so NVCC ignores it
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/Oy->")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/Oy->")

    # Bypass the strict MSVC STL vs. Intel Compiler version check
    add_compile_definitions(_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)
else()
    add_compile_options(-fno-omit-frame-pointer)
endif()

include(FortranCInterface)
FortranCInterface_VERIFY(CXX)

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
# Find CUDA toolkit (This will now rigidly use the 12.6 path defined above)
find_package(CUDAToolkit REQUIRED)

# Discover Intel oneAPI compiler lib dirs dynamically (no hardcoded version number)
file(GLOB _INTEL_COMPILER_LIB_DIRS
        "/opt/intel/oneapi/compiler/*/lib"      # system-wide install (Linux standard)
        "$ENV{HOME}/intel/oneapi/compiler/*/lib" # user-local install
        "$ENV{ONEAPI_ROOT}/compiler/*/lib"       # if env var is set, also check it
        "$ENV{ONEAPI_ROOT}/compiler/latest/lib"
)
list(SORT _INTEL_COMPILER_LIB_DIRS ORDER DESCENDING) # prefer newest version first

find_library(INTLC_LIB
        NAMES intlc
        HINTS ${_INTEL_COMPILER_LIB_DIRS}
)
if(INTLC_LIB)
    get_filename_component(INTLC_LIB_DIR "${INTLC_LIB}" DIRECTORY)
    message(STATUS "Found libintlc: ${INTLC_LIB}")
else()
    message(WARNING "libintlc.so not found — run_unit_tests may fail at runtime.")
endif()

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
# Defined early so they're available to target_include_directories' INSTALL_INTERFACE below.
set(INSTALL_LIB_DIR lib)
set(INSTALL_MOD_DIR include/fortran_modules)

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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/shroud_project>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/modules>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:${INSTALL_MOD_DIR}>
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
# RPATH configuration
# =========================
# Force CMake to embed the library paths used during linking into the final executable
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# =========================
# Unit Test Executable
# =========================
enable_testing()

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/Tests/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Tests/*.cu"
)
list(FILTER TEST_SOURCES EXCLUDE REGEX "BenchMark\\.cpp$")

add_executable(run_unit_tests ${TEST_SOURCES})

set_target_properties(run_unit_tests PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared
        BUILD_RPATH "${INTLC_LIB_DIR}"
)
target_link_libraries(run_unit_tests
        PRIVATE
        CudaBandedLib
        GTest::gtest
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(run_unit_tests DISCOVERY_MODE PRE_TEST)

# =========================
# Benchmark Executable
# =========================
add_executable(run_benchmarks
        "${CMAKE_CURRENT_SOURCE_DIR}/Tests/BenchMark.cpp"
)

set_target_properties(run_benchmarks PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared
        BUILD_RPATH "${INTLC_LIB_DIR}"
)
target_link_libraries(run_benchmarks
        PRIVATE
        CudaBandedLib
        GTest::gtest
        GTest::gtest_main
)

if(NOT MSVC)
    target_link_options(run_benchmarks PRIVATE
            "-Wl,--no-undefined"
            "-Wl,--disable-new-dtags"
    )
endif()
# =========================
# Install the library and Fortran modules for external projects
# =========================
install(TARGETS CudaBandedLib
        EXPORT CudaBandedLibTargets
        ARCHIVE DESTINATION ${INSTALL_LIB_DIR}
        LIBRARY DESTINATION ${INSTALL_LIB_DIR}
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules/
        DESTINATION ${INSTALL_MOD_DIR}
        FILES_MATCHING PATTERN "*.mod"
)

if(NOT MSVC)
    target_link_options(run_unit_tests PRIVATE
            "-Wl,--no-undefined"
            "-Wl,--disable-new-dtags"   # Use DT_RPATH (propagates transitively) not DT_RUNPATH
    )
endif()

# =========================
# Export package config so downstream projects can do:
#   find_package(CudaBandedLib REQUIRED)
#   target_link_libraries(<target> PRIVATE CudaBandedLib::CudaBandedLib)
# =========================
include(CMakePackageConfigHelpers)

set(INSTALL_CONFIG_DIR ${INSTALL_LIB_DIR}/cmake/CudaBandedLib)

install(EXPORT CudaBandedLibTargets
        FILE CudaBandedLibTargets.cmake
        NAMESPACE CudaBandedLib::
        DESTINATION ${INSTALL_CONFIG_DIR}
)

configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CudaBandedLibConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/CudaBandedLibConfig.cmake"
        INSTALL_DESTINATION ${INSTALL_CONFIG_DIR}
        PATH_VARS INSTALL_LIB_DIR INSTALL_MOD_DIR
)

write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/CudaBandedLibConfigVersion.cmake"
        VERSION 1.0.0
        COMPATIBILITY SameMajorVersion
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/CudaBandedLibConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/CudaBandedLibConfigVersion.cmake"
        DESTINATION ${INSTALL_CONFIG_DIR}
)