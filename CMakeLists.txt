# SPDX-License-Identifier: MIT
# TETSUO GPU Miner - CMake Build Configuration
# Copyright (c) 2024-2026 TETSUO Contributors

cmake_minimum_required(VERSION 3.24)
project(tetsuo-gpu-miner
    VERSION 2.0.0
    DESCRIPTION "High-performance CUDA GPU miner for TETSUO blockchain"
    LANGUAGES CXX CUDA
)

#==============================================================================
# C++ and CUDA Standards
#==============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#==============================================================================
# Dependencies
#==============================================================================

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# libcurl for RPC
find_package(CURL REQUIRED)

# OpenSSL for SHA256
find_package(OpenSSL REQUIRED)

# nlohmann/json via FetchContent
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(json)

#==============================================================================
# CUDA Architecture Detection
#==============================================================================

# Supported architectures:
#   - Ampere: 80 (A100), 86 (RTX 30xx)
#   - Ada/Lovelace: 89 (RTX 40xx)
#   - Hopper: 90 (H100)
#   - Blackwell: 100, 120 (B100, B200)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;100;120" CACHE STRING "CUDA architectures")

#==============================================================================
# Common Library
#==============================================================================

add_library(tetsuo-common STATIC
    src/utils.cpp
    src/rpc.cpp
    src/mining.cpp
)

target_include_directories(tetsuo-common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CURL_INCLUDE_DIRS}
)

target_link_libraries(tetsuo-common PUBLIC
    nlohmann_json::nlohmann_json
    ${CURL_LIBRARIES}
    OpenSSL::Crypto
)

target_compile_features(tetsuo-common PUBLIC cxx_std_17)

#==============================================================================
# GPU Miner Executable
#==============================================================================

add_executable(tetsuo-miner
    src/main.cu
)

target_include_directories(tetsuo-miner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tetsuo-miner PRIVATE
    tetsuo-common
    CUDA::cudart
    pthread
)

target_compile_options(tetsuo-miner PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -Xptxas=-v
        --expt-relaxed-constexpr
    >
)

#==============================================================================
# Test Suite
#==============================================================================

# Correctness tests
add_executable(test-sha256
    tests/test_sha256.cu
)

target_include_directories(test-sha256 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test-sha256 PRIVATE
    CUDA::cudart
)

target_compile_options(test-sha256 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
    >
)

# Performance benchmarks
add_executable(benchmark
    tests/benchmark.cu
)

target_include_directories(benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(benchmark PRIVATE
    CUDA::cudart
)

target_compile_options(benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
    >
)

# Comprehensive benchmarks for all variants
add_executable(benchmark-all
    tests/benchmark_all.cu
)

target_include_directories(benchmark-all PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(benchmark-all PRIVATE
    CUDA::cudart
)

target_compile_options(benchmark-all PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
    >
)

#==============================================================================
# Installation
#==============================================================================

install(TARGETS tetsuo-miner DESTINATION bin)

# Install headers for potential library users
install(DIRECTORY include/tetsuo DESTINATION include)
install(FILES include/sha256.cuh DESTINATION include)

#==============================================================================
# Summary
#==============================================================================

message(STATUS "")
message(STATUS "TETSUO GPU Miner v${PROJECT_VERSION}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
