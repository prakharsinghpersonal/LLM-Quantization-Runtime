cmake_minimum_required(VERSION 3.18)
project(LLM-Quantization-Runtime languages CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

include_directories(src)

# Define library
add_library(LLMRuntimeLib
    src/runtime.cpp
    src/kernels/quantization.cu
)

# Link CUDA
target_link_libraries(LLMRuntimeLib
    CUDA::cudart
)

# Executable
add_executable(llm-runtime src/runtime.cpp)
target_link_libraries(llm-runtime LLMRuntimeLib)

# Python bindings would go here
