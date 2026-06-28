cmake_minimum_required(VERSION 3.18)
project(FortranCudaProject LANGUAGES Fortran CXX)

# -----------------------------
# 1. Find Dependencies
# -----------------------------
find_package(CUDAToolkit REQUIRED)

# -----------------------------
# 2. Paths & Library Discovery
# -----------------------------
set(DEP_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cmake-build-debug" CACHE PATH "Path to CudaBandedLib")

find_library(CUDA_LIB
        NAMES CudaBandedLib
        PATHS ${DEP_LIB_DIR}
        NO_DEFAULT_PATH
)

if(NOT CUDA_LIB)
    message(FATAL_ERROR "CudaBandedLib not found! Please set DEP_LIB_DIR to the folder containing the library.")
endif()

set(FORTRAN_MODULE_DIR "${DEP_LIB_DIR}/modules")

# -----------------------------
# 2b. Find Intel runtime libs
# -----------------------------
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
    message(WARNING "libintlc.so not found — fortran_app may fail at runtime.")
endif()

# -----------------------------
# 2c. Find libstdc++
# -----------------------------
if(NOT MSVC)
    execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
            OUTPUT_VARIABLE STDCXX_LIB
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT EXISTS "${STDCXX_LIB}")
        message(FATAL_ERROR "libstdc++ not found! (searched via ${CMAKE_CXX_COMPILER} -print-file-name)")
    endif()
    message(STATUS "Found libstdc++: ${STDCXX_LIB}")
endif()
# -----------------------------
# 3. Sources and Executable
# -----------------------------
add_executable(fortran_app main.f90)

target_include_directories(fortran_app PRIVATE
        ${FORTRAN_MODULE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)

# -----------------------------
# 4. Link libraries
# -----------------------------
target_link_libraries(fortran_app PRIVATE
        ${CUDA_LIB}
        CUDA::cublas
        CUDA::cusolver
        CUDA::cusparse
        CUDA::cudart
        ${STDCXX_LIB}
)

# -----------------------------
# 5. Target Properties
# -----------------------------
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set_target_properties(fortran_app PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        BUILD_RPATH "${INTLC_LIB_DIR}"
)

if(NOT MSVC)
    target_link_options(fortran_app PRIVATE
            "-Wl,--no-undefined"
            "-Wl,--disable-new-dtags"
    )
endif()