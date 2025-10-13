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

# Collect all .cu sources
file(GLOB_RECURSE SOURCES
        src/*.cu
        src/*.cpp
        src/deviceArrays/defFiles/*.cu
        src/deviceArrays/defFiles/*.cpp
)


# Exclude PoissonFDM.cu from the general source list used by the main app
list(FILTER SOURCES EXCLUDE REGEX ".*PoissonFDM\\.cu$")

# Split out library sources (everything except main.cu) for reuse in PoissonFDM target
set(LIB_SOURCES ${SOURCES})
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*main\\.cu$")

# Production executable (uses main.cu)
add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
target_link_libraries(BiCGSTAB PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)

# Test executable builds Poisson harness plus all library sources (without main.cu)
add_executable(PoissonFDM src/PoissonFDM.cu ${LIB_SOURCES})
target_include_directories(PoissonFDM PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_target_properties(PoissonFDM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(PoissonFDM
        PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
)

