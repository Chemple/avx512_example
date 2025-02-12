cmake_minimum_required(VERSION 3.10)

project(DistanceCompute)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-mavx512f" HAS_AVX512)
if(HAS_AVX512)
    message(STATUS "AVX-512 is supported by the compiler")
    add_compile_options(-mavx512f -mavx512vl -mavx512dq -mavx512bw -mavx512cd)
else()
    message(FATAL_ERROR "AVX-512 is not supported by the compiler. Please ensure your compiler supports AVX-512.")
endif()

add_subdirectory(test)
