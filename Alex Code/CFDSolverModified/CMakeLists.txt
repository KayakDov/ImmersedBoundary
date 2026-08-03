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
            $<$<COMPILE_LANGUAGE:Fortran>:-O3>
            $<$<COMPILE_LANGUAGE:Fortran>:-xHOST>
            $<$<COMPILE_LANGUAGE:Fortran>:-qmkl>
            $<$<COMPILE_LANGUAGE:Fortran>:-qopenmp>
            $<$<COMPILE_LANGUAGE:Fortran>:-heap-arrays>
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



if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM|Intel")
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    )
endif()

# ==========================================================
# CudaBandedLib — link against prebuilt library
# (modeled on fortran_app/CMakeLists.txt: no CUDA project()
#  language needed here, so nvcc never has to resolve for
#  *this* configure step — only in CudaBandedLib's own build)
# ==========================================================

# Auto-detect the driver's max supported CUDA version via `nvidia-smi`, then
# pick the newest installed toolkit that doesn't exceed it. Two install
# layouts are searched, since different machines use different install
# methods:
#   1. NVIDIA HPC SDK:        /opt/nvidia/hpc_sdk/Linux_x86_64/<sdk-ver>/cuda/<cuda-ver>
#   2. Standard CUDA toolkit: /usr/local/cuda-<cuda-ver>
# This MUST select the same toolkit version that CudaBandedLib itself was
# built against — otherwise this executable will link against a different
# major/minor cudart/cusolver/cublas than the .a file expects. This logic is
# intentionally identical to the detection block in CudaBandedLib's own
# CMakeLists.txt, so both projects converge on the same toolkit on any given
# machine without either file hardcoding a path. If neither layout is found,
# this block steps aside and lets CMake's normal find_package(CUDAToolkit)
# search take over (PATH, default system locations) rather than failing.
#
# To override manually instead (e.g. to force a specific version), configure with:
#   cmake -B build -DCUDA_TOOLKIT_VERSION=12.9
if(NOT DEFINED CUDA_TOOLKIT_VERSION)
    find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
    set(_DRIVER_MAX_CUDA "")
    if(NVIDIA_SMI_EXECUTABLE)
        execute_process(
                COMMAND ${NVIDIA_SMI_EXECUTABLE}
                OUTPUT_VARIABLE _NVIDIA_SMI_OUTPUT
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        string(REGEX MATCH "CUDA Version: ([0-9]+\\.[0-9]+)" _ "${_NVIDIA_SMI_OUTPUT}")
        set(_DRIVER_MAX_CUDA "${CMAKE_MATCH_1}")
    endif()

    if(_DRIVER_MAX_CUDA)
        message(STATUS "Driver reports max supported CUDA version: ${_DRIVER_MAX_CUDA}")
    else()
        message(STATUS "Could not detect driver's max CUDA version via nvidia-smi; will consider all installed toolkits with no ceiling.")
    endif()

    set(_CANDIDATE_VERSIONS "")
    set(_CANDIDATE_DIRS "")

    # Layout 1: NVIDIA HPC SDK, e.g. /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9
    file(GLOB _HPC_SDK_TOOLKIT_DIRS "/opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*")
    foreach(_dir ${_HPC_SDK_TOOLKIT_DIRS})
        if(IS_DIRECTORY "${_dir}")
            get_filename_component(_ver "${_dir}" NAME)
            if(_ver MATCHES "^[0-9]+\\.[0-9]+$")
                list(APPEND _CANDIDATE_VERSIONS "${_ver}")
                list(APPEND _CANDIDATE_DIRS "${_dir}")
            endif()
        endif()
    endforeach()

    # Layout 2: standard NVIDIA .deb/.run install, e.g. /usr/local/cuda-12.4
    file(GLOB _STANDARD_TOOLKIT_DIRS "/usr/local/cuda-*")
    foreach(_dir ${_STANDARD_TOOLKIT_DIRS})
        if(IS_DIRECTORY "${_dir}")
            get_filename_component(_name "${_dir}" NAME)
            string(REGEX MATCH "^cuda-([0-9]+\\.[0-9]+)$" _ "${_name}")
            if(CMAKE_MATCH_1)
                list(APPEND _CANDIDATE_VERSIONS "${CMAKE_MATCH_1}")
                list(APPEND _CANDIDATE_DIRS "${_dir}")
            endif()
        endif()
    endforeach()

    list(LENGTH _CANDIDATE_VERSIONS _CANDIDATE_COUNT)
    set(_BEST_VERSION "0.0")
    set(_BEST_DIR "")

    if(_CANDIDATE_COUNT GREATER 0)
        math(EXPR _LAST_INDEX "${_CANDIDATE_COUNT} - 1")
        foreach(_i RANGE ${_LAST_INDEX})
            list(GET _CANDIDATE_VERSIONS ${_i} _ver)
            list(GET _CANDIDATE_DIRS ${_i} _dir)
            if(_DRIVER_MAX_CUDA AND _ver VERSION_GREATER "${_DRIVER_MAX_CUDA}")
                continue()
            endif()
            if(_ver VERSION_GREATER _BEST_VERSION)
                set(_BEST_VERSION "${_ver}")
                set(_BEST_DIR "${_dir}")
            endif()
        endforeach()
    endif()

    if(_BEST_DIR)
        set(CUDA_TOOLKIT_VERSION "${_BEST_VERSION}")
        set(CUDAToolkit_ROOT "${_BEST_DIR}" CACHE PATH "Auto-selected CUDA toolkit compatible with installed driver" FORCE)
        message(STATUS "Selected CUDA toolkit: ${CUDAToolkit_ROOT}")
    elseif(_CANDIDATE_COUNT GREATER 0)
        message(FATAL_ERROR "Found installed CUDA toolkit(s) (versions: ${_CANDIDATE_VERSIONS}), but none are compatible with the driver's max CUDA version (${_DRIVER_MAX_CUDA}). Update the driver, install a compatible toolkit, or pass -DCUDA_TOOLKIT_VERSION=<version> to override.")
    else()
        message(STATUS "No CUDA toolkit found under /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/* or /usr/local/cuda-*; falling back to CMake's default CUDAToolkit search (PATH, standard system locations).")
    endif()
else()
    # Manual override path: search both known layouts for the requested version
    file(GLOB _MATCHING_DIRS
            "/opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/${CUDA_TOOLKIT_VERSION}"
            "/usr/local/cuda-${CUDA_TOOLKIT_VERSION}"
    )
    list(LENGTH _MATCHING_DIRS _MATCH_COUNT)
    if(_MATCH_COUNT EQUAL 0)
        message(FATAL_ERROR "Requested CUDA_TOOLKIT_VERSION=${CUDA_TOOLKIT_VERSION} not found under /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/ or /usr/local/cuda-${CUDA_TOOLKIT_VERSION}.")
    endif()
    list(GET _MATCHING_DIRS 0 _CHOSEN_DIR)
    set(CUDAToolkit_ROOT "${_CHOSEN_DIR}" CACHE PATH "Manually selected CUDA toolkit" FORCE)
    message(STATUS "Using manually specified CUDA toolkit: ${CUDAToolkit_ROOT}")
endif()

find_package(CUDAToolkit REQUIRED)

set(DEP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

if(EXISTS "${DEP_ROOT}/cmake-build-release/libCudaBandedLib.a")
    set(DEP_LIB_DIR "${DEP_ROOT}/cmake-build-release")
elseif(EXISTS "${DEP_ROOT}/cmake-build-debug/libCudaBandedLib.a")
    set(DEP_LIB_DIR "${DEP_ROOT}/cmake-build-debug")
else()
    set(DEP_LIB_DIR "${DEP_ROOT}/cmake-build-release")
endif()

message(STATUS "Using CudaBandedLib from ${DEP_LIB_DIR}")

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

# CudaBandedLib's Shroud-generated Fortran wrappers must NOT be compiled here.
# CudaBandedLib's own build already compiles these (or should -- see its
# CMakeLists.txt) and deposits the resulting .mod files in ${FORTRAN_MODULE_DIR}
# (set below from ${DEP_LIB_DIR}/modules), which is already added as an include
# directory for both targets. Compiling a second local copy of the same module
# here would create two independent definitions of eigenbcgsolver_eigen_mod;
# since Fortran_MODULE_DIRECTORY for THIS target points elsewhere
# (${CMAKE_CURRENT_BINARY_DIR}/modules), the locally-compiled copy would shadow
# CudaBandedLib's and silently reintroduce the stale-interface bug every time
# the two copies drift apart (e.g. after regenerating with Shroud but only
# updating one of them). Keep exactly one compiled copy of each wrapper, built
# by CudaBandedLib, and consumed here purely via ${FORTRAN_MODULE_DIR} +
# ${CUDA_BANDED_LIB}.
list(FILTER ALEX_SOURCES EXCLUDE REGEX ".*/wrapfEigenBCGSolver_eigen\\.f90$")
list(FILTER ALEX_SOURCES EXCLUDE REGEX ".*/wrapfEigenBCGSolver_ImEq\\.f90$")
list(FILTER ALEX_SOURCES EXCLUDE REGEX ".*/wrapfeigenbcgsolver\\.f90$")

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
message(STATUS "CUDA toolkit   : ${CUDAToolkit_ROOT}")
message(STATUS "===============================")
message(STATUS "")
