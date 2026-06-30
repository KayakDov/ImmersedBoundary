cmake_minimum_required(VERSION 3.23)

# =========================
# Toolchain: match Alex's original ifx + MKL build
# =========================
# Alex's Test_TPF.link_file used:
#   ifx -O3 -r8 -qmkl -qopenmp -march=core-avx2 -qopt-zmm-usage=high -ftz
# Set the Fortran compiler to Intel's ifx BEFORE project() so CMake picks it up.
#
# IDEs (CLion, etc.) often invoke cmake with a clean environment that does not
# have oneAPI's setvars.sh sourced, so a bare "ifx" can fail to resolve even
# though it works fine from an interactive shell. Search common install
# locations for a full path before falling back to the bare command name.
if(NOT DEFINED CMAKE_Fortran_COMPILER)
    file(GLOB _IFX_CANDIDATES
            "/opt/intel/oneapi/compiler/*/bin/ifx"
            "$ENV{HOME}/intel/oneapi/compiler/*/bin/ifx"
            "$ENV{ONEAPI_ROOT}/compiler/*/bin/ifx"
    )
    list(SORT _IFX_CANDIDATES ORDER DESCENDING) # prefer newest version first
    list(LENGTH _IFX_CANDIDATES _IFX_CANDIDATES_COUNT)

    if(_IFX_CANDIDATES_COUNT GREATER 0)
        list(GET _IFX_CANDIDATES 0 _IFX_FULL_PATH)
        set(CMAKE_Fortran_COMPILER "${_IFX_FULL_PATH}")
        message(STATUS "Found ifx: ${CMAKE_Fortran_COMPILER}")
    else()
        # Last resort: hope it's on PATH. If this fails, set
        # CMAKE_Fortran_COMPILER explicitly in your CMake profile, e.g.:
        #   -DCMAKE_Fortran_COMPILER=/opt/intel/oneapi/compiler/2026.0/bin/ifx
        set(CMAKE_Fortran_COMPILER ifx)
        message(WARNING "Could not locate ifx under common oneAPI install paths. "
                "If configuration fails, set CMAKE_Fortran_COMPILER explicitly to the full path to ifx.")
    endif()
endif()

# Modify the project line:
project(AlexFortranCode LANGUAGES Fortran CUDA)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# =========================
# Standards
# =========================
set(CMAKE_Fortran_STANDARD 2008)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# =========================
# Find Intel runtime libs (libintlc)
# =========================
# ifx-compiled binaries link against libimf.so, which in turn depends on
# libintlc.so.5. This isn't on the default runtime search path unless
# oneAPI's setvars.sh has been sourced, which CLion's clean environment
# doesn't do - same root cause as the ifx/MKL path issues above. Locate it
# and bake it into the executable's RPATH so it resolves at runtime
# regardless of shell environment.
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
    message(WARNING "libintlc.so not found - Test_TPF may fail to start at runtime.")
endif()

# =========================
# Locate CudaBandedLib
# =========================
# CudaBandedLib is built/installed separately from its own project
# (~/Documents/TelAvivU). Point CMake at the *install* tree, e.g.:
#   cmake -DCudaBandedLib_ROOT=~/Documents/TelAvivU/install ..
# or add that path to CMAKE_PREFIX_PATH.
find_package(CudaBandedLib REQUIRED)

# =========================
# Compiler flags (match Alex's link_file exactly)
# =========================
# -r8            : promote default real to 8 bytes
# -qmkl          : link Intel MKL (BLAS/LAPACK: DGEEV, DGETRI, DGEMM, etc.)
# -qopenmp       : OpenMP parallel regions (!$OMP directives in the .f90 files)
# -march=core-avx2, -qopt-zmm-usage=high, -ftz : Alex's original perf tuning flags
# -heap-arrays   : NOT in Alex's original flags - added to fix a stack overflow.
#                  Several subroutines (e.g. EVDmethod) declare large automatic
#                  local arrays sized by Nxsol/Nysol/Nzsol (257x257 doubles each
#                  at Nn=256), which ifx places on the stack by default. Combined
#                  with -qopenmp, each worker thread gets its own (often small)
#                  stack, and these arrays overflow it -> SIGSEGV immediately on
#                  entering the first parallel region. -heap-arrays forces
#                  automatic arrays above a given size (in KB; 0 = all of them)
#                  onto the heap instead, which sidesteps thread stack limits
#                  entirely. This is a build-time fix and works the same way
#                  regardless of launcher (CLion, terminal, etc.), unlike
#                  OMP_STACKSIZE/KMP_STACKSIZE which depend on the environment
#                  actually reaching the process at runtime.
if(CMAKE_Fortran_COMPILER_ID STREQUAL "Intel" OR CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM")
    add_compile_options(
            $<$<COMPILE_LANGUAGE:Fortran>:-r8>
            $<$<COMPILE_LANGUAGE:Fortran>:-qmkl>
            $<$<COMPILE_LANGUAGE:Fortran>:-qopenmp>
            $<$<COMPILE_LANGUAGE:Fortran>:-march=core-avx2>
            $<$<COMPILE_LANGUAGE:Fortran>:-qopt-zmm-usage=high>
            $<$<COMPILE_LANGUAGE:Fortran>:-ftz>
            $<$<COMPILE_LANGUAGE:Fortran>:-heap-arrays>
    )

    # -qmkl expands to -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core etc.,
    # but ifx only knows where to *find* those libraries if MKLROOT/
    # LIBRARY_PATH are set (normally via oneAPI's setvars.sh). CLion invokes
    # cmake in a clean environment without that sourced, so the linker fails
    # with "cannot find -lmkl_intel_lp64" even though -qmkl itself reached
    # ifx correctly. Locate MKL's lib dir explicitly and pass it via -L so
    # this doesn't depend on shell environment.
    file(GLOB _MKL_LIB_CANDIDATES
            "/opt/intel/oneapi/mkl/*/lib"
            "$ENV{HOME}/intel/oneapi/mkl/*/lib"
            "$ENV{ONEAPI_ROOT}/mkl/*/lib"
    )
    list(SORT _MKL_LIB_CANDIDATES ORDER DESCENDING)
    list(LENGTH _MKL_LIB_CANDIDATES _MKL_LIB_CANDIDATES_COUNT)

    if(_MKL_LIB_CANDIDATES_COUNT GREATER 0)
        list(GET _MKL_LIB_CANDIDATES 0 _MKL_LIB_DIR)
        message(STATUS "Found MKL lib dir: ${_MKL_LIB_DIR}")
        add_link_options(-qmkl -qopenmp "-L${_MKL_LIB_DIR}" "-Wl,-rpath,${_MKL_LIB_DIR}")
    else()
        message(WARNING "Could not locate MKL's lib directory under common oneAPI install paths. "
                "Link will likely fail with 'cannot find -lmkl_*'. "
                "Either source oneAPI's setvars.sh before launching CLion, or set "
                "MKL lib path manually via -DCMAKE_Fortran_LINK_FLAGS=-L<path-to-mkl-lib>.")
        add_link_options(-qmkl -qopenmp)
    endif()
else()
    message(WARNING "Fortran compiler is not Intel ifx/ifort - MKL flags (-qmkl) will not apply. "
            "Set CMAKE_Fortran_COMPILER=ifx and reconfigure for an exact match to Alex's original build.")
endif()

# =========================
# Source collection
# =========================
# modfv_3D.f90 defines the modules (Size, Numbers, Grid, Variables,
# EVD_Operators, Thomas_coefficients) that the rest of the files USE.
# CMake's Fortran dependency scanner orders compilation automatically,
# so a plain glob is sufficient - no manual ordering needed.
file(GLOB_RECURSE ALEX_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.f90"
)

add_executable(Test_TPF ${ALEX_SOURCES})

set_target_properties(Test_TPF PROPERTIES
        Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules
        BUILD_RPATH "${INTLC_LIB_DIR}"
)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

target_link_libraries(Test_TPF
        PRIVATE
        CudaBandedLib::CudaBandedLib
        CUDA::cudart       # Add this for core CUDA runtime
        CUDA::cusolver
        CUDA::cublas
        CUDA::cusparse
        stdc++
        stdc++fs
)

if(NOT MSVC)
    # CRITICAL: -Wl,--disable-new-dtags forces the linker to emit the
    # old-style DT_RPATH tag instead of the modern default DT_RUNPATH.
    # DT_RUNPATH is only consulted by the dynamic linker when resolving the
    # executable's *direct* dependencies; it is explicitly NOT propagated
    # when resolving the dependencies of those dependencies. libintlc.so.5
    # is a transitive dependency here (Test_TPF -> libimf.so -> libintlc.so.5,
    # via ifx's runtime), so with RUNPATH semantics the search path set on
    # Test_TPF never gets consulted when libimf.so itself looks for
    # libintlc.so.5, and the load fails at startup. DT_RPATH propagates
    # transitively and fixes this. (Same fix CudaBandedLib's own
    # CMakeLists.txt already applies to its test/benchmark executables.)
    target_link_options(Test_TPF PRIVATE
            "-Wl,--disable-new-dtags"
    )
endif()