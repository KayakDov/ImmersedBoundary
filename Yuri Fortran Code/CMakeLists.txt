cmake_minimum_required(VERSION 3.23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------------------------------------
# Pick Intel Fortran compiler if CMake was not told already
# ----------------------------------------------------------
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

project(YuriLid3D LANGUAGES Fortran)

set(CMAKE_Fortran_STANDARD 2008)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# ----------------------------------------------------------
# External libraries
# ----------------------------------------------------------
set(MUMPS_ROOT "$ENV{HOME}/fastspace/MUMPS_5.1.Jun2017_SA")
set(DFFTPACK_ROOT "$ENV{HOME}/fastspace/dfftpack-master")

file(GLOB _INTEL_LIB_DIRS
    "/opt/intel/oneapi/compiler/*/lib"
    "$ENV{HOME}/intel/oneapi/compiler/*/lib"
    "$ENV{ONEAPI_ROOT}/compiler/*/lib"
    "$ENV{ONEAPI_ROOT}/compiler/latest/lib"
)
list(SORT _INTEL_LIB_DIRS ORDER DESCENDING)

find_library(INTLC_LIB
    NAMES intlc
    HINTS ${_INTEL_LIB_DIRS}
)

file(GLOB _MKL_LIB_DIRS
    "/opt/intel/oneapi/mkl/*/lib"
    "$ENV{HOME}/intel/oneapi/mkl/*/lib"
    "$ENV{ONEAPI_ROOT}/mkl/*/lib"
)
list(SORT _MKL_LIB_DIRS ORDER DESCENDING)
list(GET _MKL_LIB_DIRS 0 MKL_LIB_DIR)

find_library(MKL_INTEL_LP64 NAMES mkl_intel_lp64 HINTS "${MKL_LIB_DIR}")
find_library(MKL_INTEL_THREAD NAMES mkl_intel_thread HINTS "${MKL_LIB_DIR}")
find_library(MKL_CORE        NAMES mkl_core        HINTS "${MKL_LIB_DIR}")
find_library(IOMP5_LIB       NAMES iomp5           HINTS ${_INTEL_LIB_DIRS})

# ----------------------------------------------------------
# Sources
# ----------------------------------------------------------
file(GLOB SOLVER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/Source/*.f90"
)

add_executable(SchurSR ${SOLVER_SOURCES})

set_target_properties(SchurSR PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)

target_include_directories(SchurSR PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/modules"
    "${MUMPS_ROOT}/include"
)

# ----------------------------------------------------------
# Intel-specific compile flags
# ----------------------------------------------------------
if(CMAKE_Fortran_COMPILER_ID STREQUAL "Intel"
   OR CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(SchurSR PRIVATE
        -O3
        -mcmodel=large
        -assume buffered_io
        -qopenmp
        -parallel
    )
endif()

find_package(OpenMP REQUIRED COMPONENTS Fortran)
find_package(Threads REQUIRED)

# ----------------------------------------------------------
# Link libraries
# ----------------------------------------------------------
target_link_libraries(SchurSR PRIVATE
    OpenMP::OpenMP_Fortran
    "${MUMPS_ROOT}/lib/libdmumps.a"
    "${MUMPS_ROOT}/lib/libmumps_common.a"
    "${MUMPS_ROOT}/lib/libzmumps.a"
    "${DFFTPACK_ROOT}/libdfftpack.a"
    "${MUMPS_ROOT}/PORD/lib/libpord.a"
    "${MUMPS_ROOT}/libseq/libmpiseq.a"
    ${MKL_INTEL_LP64}
    ${MKL_INTEL_THREAD}
    ${MKL_CORE}
    ${IOMP5_LIB}
    Threads::Threads
)

if(INTLC_LIB)
    set_target_properties(SchurSR PROPERTIES
        BUILD_RPATH "${_INTEL_LIB_DIRS};${MKL_LIB_DIR}"
    )
endif()
