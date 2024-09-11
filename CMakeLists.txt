cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gmap VERSION 1.0 LANGUAGES CXX)

option(GMAP_BUILD_EXERCISES "Build exercises" ON)
option(GMAP_BUILD_CUDA "Build CUDA code" OFF)

if (GMAP_BUILD_CUDA)
    enable_language(CUDA)
endif()

if (GMAP_BUILD_CUDA)
    add_executable(App src/main.cu)
    set_target_properties(App PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_COMPULE_FEATURES cuda_std_17
        COMPILE_WARNINGS_AS_ERRORS ON
        CUDA_ARCHITECTURES 50
    )
endif()

if (GMAP_BUILD_EXERCISES)
    if (GMAP_BUILD_CUDA)
        add_executable(first-cuda-program exercises/first-cuda-program.cu)
        set_target_properties(first-cuda-program PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_COMPULE_FEATURES cuda_std_17
            COMPILE_WARNINGS_AS_ERRORS ON
            CUDA_ARCHITECTURES 50
        )
    endif()
    add_executable(openmp-hello exercises/openmp-hello.cpp)
    set_target_properties(openmp-hello PROPERTIES
        COMPILE_FEATURES cxx_std_17
        COMPILE_WARNINGS_AS_ERRORS ON
    )
    target_compile_options(openmp-hello PRIVATE -fopenmp)
    target_link_libraries(openmp-hello PRIVATE -fopenmp)

    add_executable(addition exercises/addition.cpp)
    set_target_properties(addition PROPERTIES
        COMPILE_FEATURES cxx_std_17
        COMPILE_WARNINGS_AS_ERRORS ON
    )
    target_compile_options(addition PRIVATE -fopenmp)
    target_link_libraries(addition PRIVATE -fopenmp)
endif()

