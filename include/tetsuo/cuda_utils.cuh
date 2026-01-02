// SPDX-License-Identifier: MIT
// TETSUO GPU Miner - CUDA Utilities
// Copyright (c) 2024-2026 TETSUO Contributors

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace tetsuo {

/// CUDA error checking macro
///
/// Usage:
///   CUDA_CHECK(cudaMalloc(&ptr, size));
///   CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
///
/// On error, prints file/line info and exits with failure status.
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/// Check for kernel launch errors
///
/// Usage:
///   my_kernel<<<grid, block>>>(args);
///   CUDA_CHECK_KERNEL();
#define CUDA_CHECK_KERNEL() do { \
    CUDA_CHECK(cudaGetLastError()); \
    CUDA_CHECK(cudaDeviceSynchronize()); \
} while(0)

/// Get optimal grid size for device
/// @param device_id CUDA device ID
/// @param block_size Thread block size
/// @return Recommended grid size
inline int get_optimal_grid_size(int device_id, int block_size) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    // Use 256 blocks per SM for good occupancy
    return prop.multiProcessorCount * 256;
}

}  // namespace tetsuo
