cmake_minimum_required(VERSION 3.23)

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

project(CFDSolverOrig LANGUAGES Fortran)

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

        $<$<COMPILE_LANGUAGE:Fortran>:-check=all>

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
# Source files
# ==========================================================

file(GLOB ALEX_SOURCES CONFIGURE_DEPENDS

    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.f90"

    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.f"

)
list(FILTER ALEX_SOURCES
        EXCLUDE
        REGEX ".*/Mesh_Intrp\\.f90$")
# ==========================================================
# Executable
# ==========================================================

add_executable(CFDSolver

    ${ALEX_SOURCES}

)

# ==========================================================
# Module output
# ==========================================================

set_target_properties(CFDSolver PROPERTIES

    Fortran_MODULE_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}/modules

)

# ==========================================================
# Runtime search path
# ==========================================================

if(INTLC_LIB_DIR)

    set_target_properties(CFDSolver PROPERTIES

        BUILD_RPATH "${INTLC_LIB_DIR}"

    )

endif()

if(NOT MSVC)

    target_link_options(CFDSolver PRIVATE

        "-Wl,--disable-new-dtags"

    )

endif()

message(STATUS "")
message(STATUS "===============================")
message(STATUS "Intel compiler : ${CMAKE_Fortran_COMPILER}")
message(STATUS "MKL library    : ${MKL_LIB_DIR}")
message(STATUS "Intel runtime  : ${INTLC_LIB_DIR}")
message(STATUS "===============================")
message(STATUS "")
