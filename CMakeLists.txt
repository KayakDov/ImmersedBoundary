cmake_minimum_required(VERSION 3.24)

# Add the correct library path to the global linker flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/cuda/lib64")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L/usr/local/cuda/lib64")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()


project(BiCGSTAB LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find Packages ---

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
# Add the correct CUDA 13.0 library path to the linker search paths
link_directories(/usr/local/cuda/lib64)
# --- 1. Gather all source files for the library ---

file(GLOB_RECURSE LIB_SOURCES
        src/*.cu
        src/*.cpp
        src/deviceArrays/defFiles/*.cu
        src/deviceArrays/defFiles/*.cpp
)

# Exclude all files with main functions from the library

list(FILTER LIB_SOURCES EXCLUDE REGEX ".*main\.cu$")
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*RunDirectSolver\.cu$")
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*FortranBindings\.cu$")

# --- 2. Create shared static library ---

add_library(BiCGSTAB_LIB STATIC
        ${LIB_SOURCES}
        src/deviceArrays/defFiles/Streamable.cpp
        src/deviceArrays/headers/Streamable.h
        src/Poisson/DirectSolver.cu
        src/Poisson/EigenDecompSolver.h
        src/Poisson/CubeBoundary.h
)
target_include_directories(BiCGSTAB_LIB PUBLIC ${CMAKE_SOURCE_DIR}/src)
set_target_properties(BiCGSTAB_LIB PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA libraries

target_link_libraries(BiCGSTAB_LIB PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        # Explicitly link the missing library using its absolute path
        "/usr/local/cuda/targets/x86_64-linux/lib/libculibos.a"
)

# --- 3. Main Executable: BiCGSTAB ---

add_executable(BiCGSTAB src/main.cu)
target_include_directories(BiCGSTAB PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(BiCGSTAB PRIVATE BiCGSTAB_LIB Threads::Threads)
set_target_properties(BiCGSTAB PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(BiCGSTAB PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo -Xcompiler -fno-inline>
)

# --- 4. Test Executable: FortranBindings ---

add_executable(EigenDecomp src/Poisson/FortranBindings.cu)
target_include_directories(EigenDecomp PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(EigenDecomp PRIVATE BiCGSTAB_LIB)
set_target_properties(EigenDecomp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --- 5. Test Executable: DirectSolver ---

add_executable(DirectSolver src/BiCGSTAB/RunDirectSolver.cu)
target_include_directories(DirectSolver PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(DirectSolver PRIVATE BiCGSTAB_LIB)
set_target_properties(DirectSolver PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
