cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gmap VERSION 1.0 LANGUAGES CXX CUDA)

add_executable(App src/main.cu)
set_target_properties(App PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_COMPULE_FEATURES cuda_std_17
    COMPILE_WARNINGS_AS_ERRORS ON
    CUDA_ARCHITECTURES 50
)

add_executable(first-cuda-program exercises/first-cuda-program.cu)
set_target_properties(first-cuda-program PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_COMPULE_FEATURES cuda_std_17
    COMPILE_WARNINGS_AS_ERRORS ON
    CUDA_ARCHITECTURES 50
)
