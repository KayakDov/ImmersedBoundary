cmake_minimum_required(VERSION 3.24)

# Work around CLion/CMake setting CMAKE_CUDA_ARCHITECTURES to an empty value.
# Set a sensible default before enabling the CUDA language to avoid configuration errors.
# "native" lets the compiler target the host GPU; replace with specific SMs if needed, e.g. 52;70;86
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(BiCGSTAB LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find Packages ---
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# --- Library Source Collection ---
# Collect all .cu and .cpp sources recursively.
file(GLOB_RECURSE ALL_PROJECT_SOURCES
        src/*.cu
        src/*.cpp
        src/deviceArrays/defFiles/*.cu
        src/deviceArrays/defFiles/*.cpp
)

# Filter out *all* main entry points from the shared library list.
# These files will be added only to their respective executable targets.
list(FILTER ALL_PROJECT_SOURCES EXCLUDE REGEX ".*main\\.cu$")
list(FILTER ALL_PROJECT_SOURCES EXCLUDE REGEX ".*FiniteDiffMethodPoisson\\.cu$")
list(FILTER ALL_PROJECT_SOURCES EXCLUDE REGEX ".*FastDiagonalizationMethod\\.cu$")

# --- 1. Shared Static Library Target ---
# Create a library from all shared code that does NOT contain a main() function.
add_library(BiCGSTAB_LIB STATIC ${ALL_PROJECT_SOURCES})
target_include_directories(BiCGSTAB_LIB PUBLIC ${CMAKE_SOURCE_DIR}/src)

# Configure CUDA for the shared library
set_target_properties(BiCGSTAB_LIB PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA/cuBLAS/cuSOLVER to the shared library
target_link_libraries(BiCGSTAB_LIB
        PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
)

# --- 2. Production Executable (BiCGSTAB) ---
# Use only 'main.cu' and link the shared library.
add_executable(${PROJECT_NAME} src/main.cu)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME} PRIVATE BiCGSTAB_LIB Threads::Threads)

# Set specific production/debugging compile options
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo -Xcompiler -fno-inline>
)

# --- 3. Test Executable (PoissonFDM) ---
# Use only 'src/Poisson/FiniteDiffMethodPoisson.cu' and link the shared library.
add_executable(PoissonFDM src/Poisson/FiniteDiffMethodPoisson.cu)
target_include_directories(PoissonFDM PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(PoissonFDM PRIVATE BiCGSTAB_LIB)

# NOTE: The CUDA_SEPARABLE_COMPILATION property is effectively managed by the linked library (BiCGSTAB_LIB)
set_target_properties(PoissonFDM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# --- 4. Test Executable (DiagnolizeMethod) ---
# Use only 'src/Poisson/FastDiagonalizationMethod.cu' and link the shared library.
add_executable(DiagnolizeMethod src/Poisson/FastDiagonalizationMethod.cu)
target_include_directories(DiagnolizeMethod PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(DiagnolizeMethod PRIVATE BiCGSTAB_LIB)

# NOTE: The CUDA_SEPARABLE_COMPILATION property is effectively managed by the linked library (BiCGSTAB_LIB)
set_target_properties(DiagnolizeMethod PROPERTIES CUDA_SEPARABLE_COMPILATION ON)