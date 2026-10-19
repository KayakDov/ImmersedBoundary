cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0135 NEW)

# =========================
# Hardware / Toolkit Alignment
# =========================
if(DEFINED CUDAToolkit_ROOT AND NOT CUDAToolkit_ROOT STREQUAL "")
    message(STATUS "Using user-provided CUDAToolkit_ROOT: ${CUDAToolkit_ROOT}")
    if(EXISTS "${CUDAToolkit_ROOT}/bin/nvcc.exe")
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc.exe" CACHE FILEPATH "nvcc" FORCE)
    elseif(EXISTS "${CUDAToolkit_ROOT}/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc" CACHE FILEPATH "nvcc" FORCE)
    endif()
else()
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

        # Layout 2: Standard NVIDIA installs (Linux and Windows)
        file(GLOB _STANDARD_TOOLKIT_DIRS
                "/usr/local/cuda-*"
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*"
        )
        foreach(_dir ${_STANDARD_TOOLKIT_DIRS})
            if(IS_DIRECTORY "${_dir}")
                get_filename_component(_name "${_dir}" NAME)
                if(_name MATCHES "^cuda-([0-9]+\\.[0-9]+)$" OR _name MATCHES "^v([0-9]+\\.[0-9]+)$")
                    list(APPEND _CANDIDATE_VERSIONS "${CMAKE_MATCH_1}")
                    list(APPEND _CANDIDATE_DIRS "${_dir}")
                endif()
            endif()
        endforeach()

        # Layout 3: User-local or environment-defined toolkit (including GitHub Actions CUDA_PATH)
        if(DEFINED ENV{CUDA_PATH} AND IS_DIRECTORY "$ENV{CUDA_PATH}")
            list(APPEND _CANDIDATE_DIRS "$ENV{CUDA_PATH}")
            if("$ENV{CUDA_PATH}" MATCHES "v([0-9]+\\.[0-9]+)")
                list(APPEND _CANDIDATE_VERSIONS "${CMAKE_MATCH_1}")
            else()
                list(APPEND _CANDIDATE_VERSIONS "12.6")
            endif()
        endif()

        file(GLOB _USER_LOCAL_TOOLKIT_DIRS "$ENV{HOME}/cuda-*")
        foreach(_dir ${_USER_LOCAL_TOOLKIT_DIRS})
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
            if(EXISTS "${_BEST_DIR}/bin/nvcc")
                set(CMAKE_CUDA_COMPILER "${_BEST_DIR}/bin/nvcc" CACHE FILEPATH "Auto-selected nvcc matching CUDAToolkit_ROOT" FORCE)
            elseif(EXISTS "${_BEST_DIR}/bin/nvcc.exe")
                set(CMAKE_CUDA_COMPILER "${_BEST_DIR}/bin/nvcc.exe" CACHE FILEPATH "Auto-selected nvcc matching CUDAToolkit_ROOT" FORCE)
            endif()
            message(STATUS "Selected CUDA toolkit: ${CUDAToolkit_ROOT}")
            message(STATUS "Selected CUDA compiler: ${CMAKE_CUDA_COMPILER}")
        elseif(_CANDIDATE_COUNT GREATER 0)
            message(FATAL_ERROR "Found installed CUDA toolkit(s) (versions: ${_CANDIDATE_VERSIONS}), but none are compatible with the driver's max CUDA version (${_DRIVER_MAX_CUDA}). Update the driver, install a compatible toolkit, or pass -DCUDA_TOOLKIT_VERSION=<version> to override.")
        else()
            message(STATUS "No CUDA toolkit found under known paths; falling back to CMake's default CUDAToolkit search.")
        endif()
    else()
        # Manual override path: search all known layouts for the requested version (Linux and Windows)
        file(GLOB _MATCHING_DIRS
                "/opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/${CUDA_TOOLKIT_VERSION}"
                "/usr/local/cuda-${CUDA_TOOLKIT_VERSION}"
                "$ENV{HOME}/cuda-${CUDA_TOOLKIT_VERSION}"
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_TOOLKIT_VERSION}"
        )
        if(DEFINED ENV{CUDA_PATH} AND "$ENV{CUDA_PATH}" MATCHES "v?${CUDA_TOOLKIT_VERSION}")
            list(APPEND _MATCHING_DIRS "$ENV{CUDA_PATH}")
        endif()
        list(LENGTH _MATCHING_DIRS _MATCH_COUNT)
        if(_MATCH_COUNT EQUAL 0)
            message(FATAL_ERROR "Requested CUDA_TOOLKIT_VERSION=${CUDA_TOOLKIT_VERSION} not found under standard paths.")
        endif()
        list(GET _MATCHING_DIRS 0 _CHOSEN_DIR)
        set(CUDAToolkit_ROOT "${_CHOSEN_DIR}" CACHE PATH "Manually selected CUDA toolkit" FORCE)
        if(EXISTS "${_CHOSEN_DIR}/bin/nvcc.exe")
            set(CMAKE_CUDA_COMPILER "${_CHOSEN_DIR}/bin/nvcc.exe" CACHE FILEPATH "Manually selected nvcc matching CUDAToolkit_ROOT" FORCE)
        elseif(EXISTS "${_CHOSEN_DIR}/bin/nvcc")
            set(CMAKE_CUDA_COMPILER "${_CHOSEN_DIR}/bin/nvcc" CACHE FILEPATH "Manually selected nvcc matching CUDAToolkit_ROOT" FORCE)
        endif()
        message(STATUS "Using manually specified CUDA toolkit: ${CUDAToolkit_ROOT}")
        message(STATUS "Using manually specified CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    endif()
endif()

# ARCHITECTURE: Set this BEFORE project() to ensure it applies globally
# Using "native" avoids hardcoding to architecture 89, preventing compilation
# failures if this is run on different hardware (e.g., cluster nodes).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# =========================
# Compiler discovery
# =========================

if(UNIX AND NOT APPLE)

    # Prefer Intel C compiler if available
    if(NOT DEFINED CMAKE_C_COMPILER)
        file(GLOB _ICX_CANDIDATES
                "/opt/intel/oneapi/compiler/*/bin/icx"
                "$ENV{HOME}/intel/oneapi/compiler/*/bin/icx"
                "$ENV{ONEAPI_ROOT}/compiler/*/bin/icx"
        )
        list(SORT _ICX_CANDIDATES ORDER DESCENDING)
        list(LENGTH _ICX_CANDIDATES _ICX_COUNT)
        if(_ICX_COUNT GREATER 0)
            list(GET _ICX_CANDIDATES 0 _ICX)
            set(CMAKE_C_COMPILER "${_ICX}")
        endif()
    endif()

    # Prefer Intel C++ compiler if available
    if(NOT DEFINED CMAKE_CXX_COMPILER)
        file(GLOB _ICPX_CANDIDATES
                "/opt/intel/oneapi/compiler/*/bin/icpx"
                "$ENV{HOME}/intel/oneapi/compiler/*/bin/icpx"
                "$ENV{ONEAPI_ROOT}/compiler/*/bin/icpx"
        )
        list(SORT _ICPX_CANDIDATES ORDER DESCENDING)
        list(LENGTH _ICPX_CANDIDATES _ICPX_COUNT)
        if(_ICPX_COUNT GREATER 0)
            list(GET _ICPX_CANDIDATES 0 _ICPX)
            set(CMAKE_CXX_COMPILER "${_ICPX}")
        endif()
    endif()

    # Prefer Intel Fortran compiler if available
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
        endif()
    endif()

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
set(CMAKE_Fortran_STANDARD 2008)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# =========================
# Dependencies
# =========================
# Find CUDA toolkit (uses whichever CUDAToolkit_ROOT was auto-selected above)
find_package(CUDAToolkit REQUIRED)
# Discover NVIDIA HPC SDK CUDA math library include dirs dynamically.
set(_HAVE_CUSOLVER_ALREADY FALSE)
foreach(_inc_dir ${CUDAToolkit_INCLUDE_DIRS})
    if(EXISTS "${_inc_dir}/cusolverDn.h")
        set(_HAVE_CUSOLVER_ALREADY TRUE)
        break()
    endif()
endforeach()

if(_HAVE_CUSOLVER_ALREADY)
    message(STATUS "cusolverDn.h already provided by CUDAToolkit_INCLUDE_DIRS; skipping HPC SDK math_libs fallback")
else()
    file(GLOB _HPC_MATH_ROOTS
            "/opt/nvidia/hpc_sdk/Linux_x86_64/*/math_libs/*/targets/x86_64-linux"
    )
    list(SORT _HPC_MATH_ROOTS ORDER DESCENDING)

    foreach(_root IN LISTS _HPC_MATH_ROOTS)
        if(EXISTS "${_root}/include/cusolverDn.h")
            message(STATUS "Found NVIDIA HPC math libraries: ${_root}")
            include_directories(SYSTEM "${_root}/include")
            break()
        endif()
    endforeach()
endif()
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.17.0
        GIT_SHALLOW    TRUE
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
# cfdSolver executable
# =========================
# src/cfdSolver/main.cu is deliberately excluded from LIB_SOURCES above (a
# library can't contain a main()), but nothing previously built it into a
# runnable program. This is that program: loads a JSON config (see
# src/cfdSolver/example_config.json) and runs one simulation to completion,
# matching ConvMain_3D_Q2D.f90's standalone executable. No extra include
# dirs needed -- CfdSimulation.h's own includes ("setup/...", "physics/...")
# resolve against its own directory (src/cfdSolver/) automatically, and
# cross-module includes like "deviceArrays/headers/handle.h" come from
# CudaBandedLib's own PUBLIC include dirs via the link below.
add_executable(cfdSolver "${CMAKE_CURRENT_SOURCE_DIR}/src/cfdSolver/main.cu")

set_target_properties(cfdSolver PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared
        BUILD_RPATH "${INTLC_LIB_DIR}"
)
target_link_libraries(cfdSolver
        PRIVATE
        CudaBandedLib
)

# =========================
# RPATH configuration
# =========================
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
        "${CMAKE_CURRENT_SOURCE_DIR}/Tests/cfdSolver/*.cpp"
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
# Tests/cfdSolver/*.cpp use unqualified includes ("SimulationConfig.h",
# "physics/Operators.cuh") that only resolve with src/cfdSolver itself as a
# search root -- CudaBandedLib's own include dirs expose src/, not src/cfdSolver/.
# No extra link library needed: src/cfdSolver/*.cpp is already compiled into
# CudaBandedLib via LIB_SOURCES' recursive glob above, and these tests are
# host-only (no CUDA/GPU calls -- see verify_padding_equivalence.cpp's header).
target_include_directories(run_unit_tests
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cfdSolver"
)

include(GoogleTest)
option(SHOW_LORENZ_PORTRAIT "Show the Lorenz window during tests" ON)

if(SHOW_LORENZ_PORTRAIT)
    set(LORENZ_WINDOW_ENABLED 1)
else()
    set(LORENZ_WINDOW_ENABLED 0)
endif()

set(TEST_ENV "SHOW_LORENZ_PORTRAIT=${LORENZ_WINDOW_ENABLED}")
if(INTLC_LIB_DIR)
    list(APPEND TEST_ENV "LD_LIBRARY_PATH=${INTLC_LIB_DIR}:$ENV{LD_LIBRARY_PATH}")
endif()

gtest_discover_tests(run_unit_tests
        DISCOVERY_MODE PRE_TEST
        PROPERTIES
        ENVIRONMENT "${TEST_ENV}"
        LABELS "unit"
)

if(WIN32)
    add_custom_command(TARGET run_unit_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E
            $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:run_unit_tests>>,copy_if_different,echo>
            $<TARGET_RUNTIME_DLLS:run_unit_tests>
            $<TARGET_FILE_DIR:run_unit_tests>
            COMMAND_EXPAND_LISTS
    )
endif()

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

gtest_discover_tests(run_benchmarks
        DISCOVERY_MODE PRE_TEST
        PROPERTIES
        ENVIRONMENT "${TEST_ENV}"
        LABELS "benchmark"
)

if(WIN32)
    add_custom_command(TARGET run_benchmarks POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E
            $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:run_benchmarks>>,copy_if_different,echo>
            $<TARGET_RUNTIME_DLLS:run_benchmarks>
            $<TARGET_FILE_DIR:run_benchmarks>
            COMMAND_EXPAND_LISTS
    )
endif()

# =========================
# Intel runtime (libimf.so etc.) rpath, explicit
# =========================
# The BUILD_RPATH property set on each target above does not reliably end
# up in the final binary -- confirmed via `readelf -d cfdSolver | grep -i
# rpath` printing nothing at all, even with INTLC_LIB_DIR correctly
# resolved, which is why running cfdSolver directly (outside CTest, which
# masked this by injecting LD_LIBRARY_PATH itself via TEST_ENV) failed with
# "libimf.so: cannot open shared object file". An explicit linker rpath
# flag is harder for CMake to silently drop than the property is. Linux
# only -- Windows has no rpath concept; its runtime DLLs are copied next to
# the executable by the WIN32 TARGET_RUNTIME_DLLS steps above instead.
if(NOT WIN32 AND INTLC_LIB_DIR)
    foreach(_rpath_target run_unit_tests run_benchmarks cfdSolver)
        target_link_options(${_rpath_target} PRIVATE
                "-Wl,-rpath,${INTLC_LIB_DIR}"
                "-Wl,--disable-new-dtags"   # embed old-style DT_RPATH, not DT_RUNPATH --
                                            # DT_RUNPATH only resolves a binary's own
                                            # direct NEEDED entries and does not
                                            # propagate to libraries those libraries in
                                            # turn depend on, unlike DT_RPATH. Applying
                                            # this consistently to every target here
                                            # (not just run_unit_tests, which is where it
                                            # previously lived alone) removes that gap.
        )
    endforeach()
endif()

# =========================
# Aggregate "run everything that is a real test, never benchmarks" target
# =========================
# run_unit_tests now includes both the original CudaBandedLib suites and the
# cfdSolver suites (Tests/cfdSolver/*.cpp, folded in above) -- one binary, so
# this list currently has one entry. Add any future standalone test
# executable here and it is automatically (a) built as a dependency and
# (b) swept into the single ctest invocation below. run_benchmarks is
# deliberately never added here -- it is additionally excluded by label as a
# second, independent safeguard in case a future test target forgets to list
# itself.
set(ALL_TEST_TARGETS
        run_unit_tests
)

add_custom_target(run_all_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C $<CONFIG> -LE "benchmark"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS ${ALL_TEST_TARGETS}
        COMMENT "Building and running every non-benchmark test suite"
)

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

# =========================
# VTK phase-portrait example
# =========================
option(BUILD_LORENZ_VIEWER "Build the Lorenz phase-portrait viewer" ON)

if(BUILD_LORENZ_VIEWER)
    find_package(VTK REQUIRED COMPONENTS
            CommonCore
            CommonDataModel
            InteractionStyle
            RenderingCore
            RenderingOpenGL2
    )

    message(STATUS "Phase-portrait support: found VTK ${VTK_VERSION}")

    set(PHASE_PORTRAIT_VTK_MODULES
            VTK::CommonCore
            VTK::CommonDataModel
            VTK::InteractionStyle
            VTK::RenderingCore
            VTK::RenderingOpenGL2
    )

    target_link_libraries(CudaBandedLib PUBLIC
            ${PHASE_PORTRAIT_VTK_MODULES}
    )

    vtk_module_autoinit(
            TARGETS
            CudaBandedLib
            run_unit_tests
            run_benchmarks
            MODULES
            ${PHASE_PORTRAIT_VTK_MODULES}
    )
endif()
