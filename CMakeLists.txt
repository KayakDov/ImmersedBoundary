cmake_minimum_required(VERSION 3.24)
project(BiCGSTAB LANGUAGES CXX CUDA)

# C++ / CUDA settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

# Export compile commands (good for IDE indexing)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find packages
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Add global CUDA library path
link_directories(/usr/local/cuda/lib64)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/cuda/lib64")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L/usr/local/cuda/lib64")

# --- 1. Gather source files for the library ---
file(GLOB_RECURSE LIB_SOURCES
        src/*.cu
        src/*.cpp
        src/deviceArrays/defFiles/*.cu
        src/deviceArrays/defFiles/*.cpp
)

# Exclude files with main functions
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*main\\.cu$")
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*RunDirectSolver\\.cu$")
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*testMethods\\.cu$")

# --- 2. Create static library ---
add_library(BiCGSTAB_LIB STATIC
        ${LIB_SOURCES}
        src/FortranBindings.hpp
)

target_include_directories(BiCGSTAB_LIB PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/deviceArrays/headers
        ${CMAKE_SOURCE_DIR}/src/BiCGSTAB
        ${CMAKE_SOURCE_DIR}/src/Poisson
)

set_target_properties(BiCGSTAB_LIB PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(BiCGSTAB_LIB PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        "/usr/local/cuda/targets/x86_64-linux/lib/libculibos.a"
)

# --- 3. Executables ---
# main.cu
add_executable(BiCGSTAB src/main.cu
        src/FortranBindings.hpp)
target_include_directories(BiCGSTAB PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(BiCGSTAB PRIVATE BiCGSTAB_LIB Threads::Threads)
set_target_properties(BiCGSTAB PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(BiCGSTAB PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo -Xcompiler -fno-inline>
)

# testMethods.cu (FortranBindings)
add_executable(EigenDecomp src/testMethods.cu
        src/FortranBindings.hpp)
target_include_directories(EigenDecomp PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(EigenDecomp PRIVATE BiCGSTAB_LIB)
set_target_properties(EigenDecomp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# RunDirectSolver.cu
add_executable(DirectSolver src/BiCGSTAB/RunDirectSolver.cu
        src/FortranBindings.hpp)
target_include_directories(DirectSolver PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(DirectSolver PRIVATE BiCGSTAB_LIB)
set_target_properties(DirectSolver PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
