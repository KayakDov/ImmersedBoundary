cmake_minimum_required(VERSION 3.23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------------------------------------
# Pick Intel Fortran compiler if CMake was not told already
# ----------------------------------------------------------
if(NOT DEFINED CMAKE_Fortran_COMPILER)
    find_program(_IFX_ON_PATH NAMES ifx)

    if(_IFX_ON_PATH)
        set(CMAKE_Fortran_COMPILER "${_IFX_ON_PATH}")
    else()
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
            message(FATAL_ERROR
                "Could not find ifx on PATH or under /opt/intel/oneapi/compiler, "
                "$ENV{HOME}/intel/oneapi/compiler, or $ENV{ONEAPI_ROOT}/compiler. "
                "Source the oneAPI setvars.sh first, or install/point at oneAPI.")
        endif()
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
list(LENGTH _MKL_LIB_DIRS _MKL_LIB_DIR_COUNT)
if(_MKL_LIB_DIR_COUNT EQUAL 0)
    message(FATAL_ERROR
        "Could not find an Intel MKL lib directory under /opt/intel/oneapi/mkl, "
        "$ENV{HOME}/intel/oneapi/mkl, or $ENV{ONEAPI_ROOT}/mkl. "
        "Source the oneAPI setvars.sh, or set ONEAPI_ROOT / adjust the glob paths above.")
endif()
list(GET _MKL_LIB_DIRS 0 MKL_LIB_DIR)

find_library(MKL_INTEL_LP64 NAMES mkl_intel_lp64 HINTS "${MKL_LIB_DIR}")
find_library(MKL_INTEL_THREAD NAMES mkl_intel_thread HINTS "${MKL_LIB_DIR}")
find_library(MKL_CORE        NAMES mkl_core        HINTS "${MKL_LIB_DIR}")
find_library(IOMP5_LIB       NAMES iomp5           HINTS ${_INTEL_LIB_DIRS})

# ----------------------------------------------------------
# Sources
# ----------------------------------------------------------
# NOTE: Source/ contains a few extra .f90 files (EVD_Thomas_3D_OMP_z,
# EVD_eigenvector_ag [empty], Mesh_Intrp, check_Lid3D, divvel_3D, gradp_3D,
# pointw_velocities, vgradf_Lid3D) that are NOT part of the working build --
# Source/Makefile.backup even shows check_Lid3D.o was deliberately dropped
# from the live Makefile. We mirror the legacy Makefile's MODS+OBJS list
# exactly (and its link order: modules first) rather than globbing the
# directory, so we don't drag in dead/WIP code that was never compiling.
set(SOLVER_MODULE_SOURCES
    modfv_Lid3D_z.f90
    FVOperators.f90
    EvdProcedures.f90
    IBSetupInterpRegular.f90
    MatrixFormAndOperate.f90
)

set(SOLVER_OBJ_SOURCES
    ConvMain_Lid3D.f90
    bounds_Lid3D_z.f90
    time_step_Lid3D_z.f90
    init_Lid3D.f90
    pointw.f90
    ekinem_3D.f90
    prmesh.f90
    MeshStretch.f90
    Solution_time_Lid3D.f90
    sublid_3D.f90
    outp_Lid3D.f90
    BICGf90.f90
)

set(SOLVER_SOURCES ${SOLVER_MODULE_SOURCES} ${SOLVER_OBJ_SOURCES})
list(TRANSFORM SOLVER_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/Source/")

foreach(_src ${SOLVER_SOURCES})
    if(NOT EXISTS "${_src}")
        message(FATAL_ERROR "Expected solver source not found: ${_src}")
    endif()
endforeach()

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
# NOTE: find_package(Threads) requires C or CXX to be enabled, but this
# project only declares Fortran. The legacy build just passed -lpthread
# directly, so we do the same instead of pulling in Threads::Threads.

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
    pthread
)

if(INTLC_LIB)
    set_target_properties(SchurSR PROPERTIES
        BUILD_RPATH "${_INTEL_LIB_DIRS};${MKL_LIB_DIR}"
    )
endif()
