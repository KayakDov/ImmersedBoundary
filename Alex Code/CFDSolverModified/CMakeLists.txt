cmake_minimum_required(VERSION 3.23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==========================================================
# Intel Fortran compiler
# ==========================================================

if(NOT DEFINED CMAKE_Fortran_COMPILER)
    file(GLOB _IFX_CANDIDATES
            "/opt/intel/oneapi/compiler/*/bin/ifx"
            "$ENV{HOME}/intel/oneapi/compiler/*/bin/ifx"
            "$ENV{ONEAPI_ROOT}/compiler/*/bin/ifx"
    )

    list(SORT _IFX_CANDIDATES ORDER DESCENDING)

    list(LENGTH _IFX_CANDIDATES _IFX_COUNT)

    if(_IFX_COUNT GREATER 0)
        list(GET _IFX_CANDIDATES 0 _IFX)
        set(CMAKE_Fortran_COMPILER "${_IFX}")
    else()
        set(CMAKE_Fortran_COMPILER ifx)
    endif()
endif()

project(CFDSolverOrig LANGUAGES Fortran CXX)

# ==========================================================
# Fortran standard
# ==========================================================

set(CMAKE_Fortran_STANDARD 2008)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# ==========================================================
# Intel runtime library discovery
# ==========================================================

file(GLOB _INTEL_COMPILER_LIB_DIRS
        "/opt/intel/oneapi/compiler/*/lib"
        "$ENV{HOME}/intel/oneapi/compiler/*/lib"
        "$ENV{ONEAPI_ROOT}/compiler/*/lib"
        "$ENV{ONEAPI_ROOT}/compiler/latest/lib"
)

list(SORT _INTEL_COMPILER_LIB_DIRS ORDER DESCENDING)

find_library(INTLC_LIB
        NAMES intlc
        HINTS ${_INTEL_COMPILER_LIB_DIRS}
)

if(INTLC_LIB)
    get_filename_component(INTLC_LIB_DIR "${INTLC_LIB}" DIRECTORY)
    message(STATUS "Found libintlc: ${INTLC_LIB}")
else()
    message(WARNING "libintlc.so not found — CFDSolver/BinaryToTxt may fail at runtime.")
endif()

# ==========================================================
# Intel MKL discovery
# ==========================================================

file(GLOB _MKL_LIB_CANDIDATES
        "/opt/intel/oneapi/mkl/*/lib"
        "$ENV{HOME}/intel/oneapi/mkl/*/lib"
        "$ENV{ONEAPI_ROOT}/mkl/*/lib"
)

list(SORT _MKL_LIB_CANDIDATES ORDER DESCENDING)

list(LENGTH _MKL_LIB_CANDIDATES _MKL_COUNT)

if(_MKL_COUNT GREATER 0)
    list(GET _MKL_LIB_CANDIDATES 0 MKL_LIB_DIR)
endif()

# ==========================================================
# Compiler flags
# ==========================================================

if(CMAKE_Fortran_COMPILER_ID STREQUAL "Intel"
        OR
        CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM")

    add_compile_options(
            $<$<COMPILE_LANGUAGE:Fortran>:-g>
            $<$<COMPILE_LANGUAGE:Fortran>:-O0>
            $<$<COMPILE_LANGUAGE:Fortran>:-r8>
            $<$<COMPILE_LANGUAGE:Fortran>:-qmkl>
            $<$<COMPILE_LANGUAGE:Fortran>:-qopenmp>
            $<$<COMPILE_LANGUAGE:Fortran>:-heap-arrays>
            $<$<COMPILE_LANGUAGE:Fortran>:-traceback>
    )

    if(MKL_LIB_DIR)
        add_link_options(
                -qmkl
                -qopenmp
                "-L${MKL_LIB_DIR}"
                "-Wl,-rpath,${MKL_LIB_DIR}"
        )
    endif()

endif()

# ==========================================================
# CudaBandedLib — link against prebuilt library
# (modeled on fortran_app/CMakeLists.txt: no CUDA project()
#  language needed here, so nvcc never has to resolve for
#  *this* configure step — only in CudaBandedLib's own build)
# ==========================================================

find_package(CUDAToolkit REQUIRED)

set(DEP_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake-build-debug" CACHE PATH "Path to CudaBandedLib's build output")

find_library(CUDA_BANDED_LIB
        NAMES CudaBandedLib
        PATHS ${DEP_LIB_DIR}
        NO_DEFAULT_PATH
)

if(NOT CUDA_BANDED_LIB)
    message(FATAL_ERROR "CudaBandedLib not found in ${DEP_LIB_DIR}! Build CudaBandedLib standalone first, or set -DDEP_LIB_DIR=... to its build directory.")
endif()

message(STATUS "Found CudaBandedLib: ${CUDA_BANDED_LIB}")

set(FORTRAN_MODULE_DIR "${DEP_LIB_DIR}/modules")

# ==========================================================
# Source files
# ==========================================================

file(GLOB ALEX_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.f90"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.f"
)

# Exclude specific files from the general source list
list(FILTER ALEX_SOURCES EXCLUDE REGEX ".*/Mesh_Intrp\\.f90$")

# Create a list of common shared modules by removing the main CFD program
set(COMMON_SOURCES ${ALEX_SOURCES})
list(FILTER COMMON_SOURCES EXCLUDE REGEX ".*/ConvMain_3D_Q2D\\.f90$")

# ==========================================================
# Executables
# ==========================================================

# 1. Main CFD Solver
add_executable(CFDSolver "${CMAKE_CURRENT_SOURCE_DIR}/src/ConvMain_3D_Q2D.f90" ${COMMON_SOURCES})

# 2. Binary to Text Utility
add_executable(BinaryToTxt "${CMAKE_CURRENT_SOURCE_DIR}/Auxilery/BinaryToTxt.f90" ${COMMON_SOURCES})

# Apply properties to both targets
set_target_properties(CFDSolver BinaryToTxt PROPERTIES
        LINKER_LANGUAGE Fortran
        Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules
)

target_include_directories(CFDSolver PRIVATE
        ${FORTRAN_MODULE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_include_directories(BinaryToTxt PRIVATE
        ${FORTRAN_MODULE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(CFDSolver PRIVATE
        ${CUDA_BANDED_LIB}
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        CUDA::cusparse
        stdc++
        stdc++fs
)

target_link_libraries(BinaryToTxt PRIVATE
        ${CUDA_BANDED_LIB}
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        CUDA::cusparse
        stdc++
        stdc++fs
)

# ==========================================================
# Runtime search path
# ==========================================================

if(INTLC_LIB_DIR)
    set_target_properties(CFDSolver BinaryToTxt PROPERTIES
            BUILD_RPATH "${INTLC_LIB_DIR}"
    )
endif()

if(NOT MSVC)
    target_link_options(CFDSolver PRIVATE "-Wl,--disable-new-dtags")
    target_link_options(BinaryToTxt PRIVATE "-Wl,--disable-new-dtags")
endif()

message(STATUS "")
message(STATUS "===============================")
message(STATUS "Intel compiler : ${CMAKE_Fortran_COMPILER}")
message(STATUS "MKL library    : ${MKL_LIB_DIR}")
message(STATUS "Intel runtime  : ${INTLC_LIB_DIR}")
message(STATUS "CudaBandedLib  : ${CUDA_BANDED_LIB}")
message(STATUS "===============================")
message(STATUS "")