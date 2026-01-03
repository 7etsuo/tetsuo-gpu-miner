// SPDX-License-Identifier: MIT
// TETSUO GPU Miner - Comprehensive Optimization Benchmark
// Copyright (c) 2024-2026 TETSUO Contributors

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>

#include "sha256.cuh"
#include "sha256_optimized.cuh"
#include "sha256_variants.cuh"
#include "tetsuo/cuda_utils.cuh"

using namespace tetsuo;

//==============================================================================
// Benchmark Infrastructure
//==============================================================================

struct BenchResult {
    std::string name;
    double hash_rate_mh;
    double speedup;
};

std::vector<BenchResult> g_results;
double g_baseline_rate = 0;

//==============================================================================
// Generic Benchmark Function
//==============================================================================

template<typename KernelFunc>
double benchmark_kernel(
    KernelFunc kernel,
    const char* name,
    int grid_size,
    int block_size,
    int iterations,
    bool use_midstate = true
) {
    uint32_t header[20];
    for (int i = 0; i < 20; i++) header[i] = 0x12345678 + i;

    uint32_t midstate[8];
    if (use_midstate) {
        variants::compute_midstate_cpu(header, midstate);
    }

    uint32_t tail[4] = {header[16], header[17], header[18], header[19]};
    uint32_t target[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // Impossible target

    uint32_t* d_midstate;
    uint32_t* d_header;
    uint32_t* d_tail;
    uint32_t* d_target;
    uint32_t* d_result_nonce;
    uint32_t* d_result_found;

    CUDA_CHECK(cudaMalloc(&d_midstate, 32));
    CUDA_CHECK(cudaMalloc(&d_header, 80));
    CUDA_CHECK(cudaMalloc(&d_tail, 16));
    CUDA_CHECK(cudaMalloc(&d_target, 32));
    CUDA_CHECK(cudaMalloc(&d_result_nonce, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result_found, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_midstate, midstate, 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_header, header, 80, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tail, tail, 16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, target, 32, cudaMemcpyHostToDevice));

    // Warmup
    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_result_found, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
    kernel(d_midstate, d_tail, d_target, 0, d_result_nonce, d_result_found, grid_size, block_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    uint32_t nonce = 0;
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemcpy(d_result_found, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
        kernel(d_midstate, d_tail, d_target, nonce, d_result_nonce, d_result_found, grid_size, block_size);
        nonce += grid_size * block_size;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    uint64_t total_hashes = (uint64_t)grid_size * block_size * iterations;
    double hash_rate = (total_hashes / (elapsed_ms / 1000.0)) / 1e6;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_midstate);
    cudaFree(d_header);
    cudaFree(d_tail);
    cudaFree(d_target);
    cudaFree(d_result_nonce);
    cudaFree(d_result_found);

    return hash_rate;
}

//==============================================================================
// Kernel Launch Wrappers
//==============================================================================

void launch_baseline(const uint32_t* d_header, const uint32_t* d_tail, const uint32_t* d_target,
                     uint32_t nonce, uint32_t* d_result_nonce, uint32_t* d_result_found,
                     int grid, int block) {
    // Baseline uses full header, not midstate - need special handling
    mine_kernel<<<grid, block>>>(d_header, d_target, nonce, d_result_nonce, d_result_found);
}

void launch_midstate(const uint32_t* d_midstate, const uint32_t* d_tail, const uint32_t* d_target,
                     uint32_t nonce, uint32_t* d_result_nonce, uint32_t* d_result_found,
                     int grid, int block) {
    optimized::mine_kernel_optimized<<<grid, block>>>(d_midstate, d_tail, d_target, nonce, d_result_nonce, d_result_found);
}

void launch_lop3(const uint32_t* d_midstate, const uint32_t* d_tail, const uint32_t* d_target,
                 uint32_t nonce, uint32_t* d_result_nonce, uint32_t* d_result_found,
                 int grid, int block) {
    variants::mine_lop3<<<grid, block>>>(d_midstate, d_tail, d_target, nonce, d_result_nonce, d_result_found);
}

void launch_early_exit(const uint32_t* d_midstate, const uint32_t* d_tail, const uint32_t* d_target,
                       uint32_t nonce, uint32_t* d_result_nonce, uint32_t* d_result_found,
                       int grid, int block) {
    variants::mine_early_exit<<<grid, block>>>(d_midstate, d_tail, d_target, nonce, d_result_nonce, d_result_found);
}

void launch_sliding(const uint32_t* d_midstate, const uint32_t* d_tail, const uint32_t* d_target,
                    uint32_t nonce, uint32_t* d_result_nonce, uint32_t* d_result_found,
                    int grid, int block) {
    variants::mine_sliding<<<grid, block>>>(d_midstate, d_tail, d_target, nonce, d_result_nonce, d_result_found);
}

void launch_shared(const uint32_t* d_midstate, const uint32_t* d_tail, const uint32_t* d_target,
                   uint32_t nonce, uint32_t* d_result_nonce, uint32_t* d_result_found,
                   int grid, int block) {
    variants::mine_shared<<<grid, block>>>(d_midstate, d_tail, d_target, nonce, d_result_nonce, d_result_found);
}

void launch_precomp(const uint32_t* d_midstate, const uint32_t* d_tail, const uint32_t* d_target,
                    uint32_t nonce, uint32_t* d_result_nonce, uint32_t* d_result_found,
                    int grid, int block) {
    variants::mine_precomp<<<grid, block>>>(d_midstate, d_tail, d_target, nonce, d_result_nonce, d_result_found);
}

void launch_lop3_sliding(const uint32_t* d_midstate, const uint32_t* d_tail, const uint32_t* d_target,
                         uint32_t nonce, uint32_t* d_result_nonce, uint32_t* d_result_found,
                         int grid, int block) {
    variants::mine_lop3_sliding<<<grid, block>>>(d_midstate, d_tail, d_target, nonce, d_result_nonce, d_result_found);
}

void launch_lop3_shared(const uint32_t* d_midstate, const uint32_t* d_tail, const uint32_t* d_target,
                        uint32_t nonce, uint32_t* d_result_nonce, uint32_t* d_result_found,
                        int grid, int block) {
    variants::mine_lop3_shared<<<grid, block>>>(d_midstate, d_tail, d_target, nonce, d_result_nonce, d_result_found);
}

void launch_lop3_sliding_shared(const uint32_t* d_midstate, const uint32_t* d_tail, const uint32_t* d_target,
                                uint32_t nonce, uint32_t* d_result_nonce, uint32_t* d_result_found,
                                int grid, int block) {
    variants::mine_lop3_sliding_shared<<<grid, block>>>(d_midstate, d_tail, d_target, nonce, d_result_nonce, d_result_found);
}

void launch_all_optimizations(const uint32_t* d_midstate, const uint32_t* d_tail, const uint32_t* d_target,
                              uint32_t nonce, uint32_t* d_result_nonce, uint32_t* d_result_found,
                              int grid, int block) {
    variants::mine_all_optimizations<<<grid, block>>>(d_midstate, d_tail, d_target, nonce, d_result_nonce, d_result_found);
}

//==============================================================================
// Baseline Benchmark (special - uses full header)
//==============================================================================

double benchmark_baseline_kernel(int grid_size, int block_size, int iterations) {
    uint32_t header[20];
    for (int i = 0; i < 20; i++) header[i] = 0x12345678 + i;

    uint32_t target[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    uint32_t* d_header;
    uint32_t* d_target;
    uint32_t* d_result_nonce;
    uint32_t* d_result_found;

    CUDA_CHECK(cudaMalloc(&d_header, 80));
    CUDA_CHECK(cudaMalloc(&d_target, 32));
    CUDA_CHECK(cudaMalloc(&d_result_nonce, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result_found, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_header, header, 80, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, target, 32, cudaMemcpyHostToDevice));

    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_result_found, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
    mine_kernel<<<grid_size, block_size>>>(d_header, d_target, 0, d_result_nonce, d_result_found);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    uint32_t nonce = 0;
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemcpy(d_result_found, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
        mine_kernel<<<grid_size, block_size>>>(d_header, d_target, nonce, d_result_nonce, d_result_found);
        nonce += grid_size * block_size;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    uint64_t total_hashes = (uint64_t)grid_size * block_size * iterations;
    double hash_rate = (total_hashes / (elapsed_ms / 1000.0)) / 1e6;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_header);
    cudaFree(d_target);
    cudaFree(d_result_nonce);
    cudaFree(d_result_found);

    return hash_rate;
}

//==============================================================================
// Correctness Verification
//==============================================================================

__global__ void compute_hash_baseline_g(const uint32_t* header, uint32_t nonce, uint32_t* hash) {
    sha256_block_header(header, nonce, hash);
}

__global__ void compute_hash_lop3(const uint32_t* midstate, const uint32_t* tail, uint32_t nonce, uint32_t* hash) {
    variants::sha256_lop3(midstate, tail, nonce, hash);
}

__global__ void compute_hash_sliding(const uint32_t* midstate, const uint32_t* tail, uint32_t nonce, uint32_t* hash) {
    variants::sha256_sliding(midstate, tail, nonce, hash);
}

__global__ void compute_hash_precomp(const uint32_t* midstate, const uint32_t* tail, uint32_t nonce, uint32_t* hash) {
    variants::sha256_precomp(midstate, tail, nonce, hash);
}

__global__ void compute_hash_lop3_sliding(const uint32_t* midstate, const uint32_t* tail, uint32_t nonce, uint32_t* hash) {
    variants::sha256_lop3_sliding(midstate, tail, nonce, hash);
}

bool verify_all_variants() {
    printf("[VERIFY] Checking all variants produce correct hashes...\n");

    srand(12345);

    uint32_t header[20];
    for (int i = 0; i < 20; i++) header[i] = rand();

    uint32_t midstate[8];
    variants::compute_midstate_cpu(header, midstate);
    uint32_t tail[4] = {header[16], header[17], header[18], header[19]};

    uint32_t* d_header;
    uint32_t* d_midstate;
    uint32_t* d_tail;
    uint32_t* d_hash_ref;
    uint32_t* d_hash_test;

    CUDA_CHECK(cudaMalloc(&d_header, 80));
    CUDA_CHECK(cudaMalloc(&d_midstate, 32));
    CUDA_CHECK(cudaMalloc(&d_tail, 16));
    CUDA_CHECK(cudaMalloc(&d_hash_ref, 32));
    CUDA_CHECK(cudaMalloc(&d_hash_test, 32));

    CUDA_CHECK(cudaMemcpy(d_header, header, 80, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_midstate, midstate, 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tail, tail, 16, cudaMemcpyHostToDevice));

    const char* variant_names[] = {"LOP3", "Sliding", "Precomp", "LOP3+Sliding"};
    void (*variant_kernels[])(const uint32_t*, const uint32_t*, uint32_t, uint32_t*) = {
        compute_hash_lop3, compute_hash_sliding, compute_hash_precomp, compute_hash_lop3_sliding
    };
    int num_variants = 4;

    bool all_pass = true;

    for (int v = 0; v < num_variants && all_pass; v++) {
        bool variant_pass = true;

        for (int iter = 0; iter < 100 && variant_pass; iter++) {
            uint32_t nonce = rand();

            // Reference hash
            compute_hash_baseline_g<<<1, 1>>>(d_header, nonce, d_hash_ref);
            // Test hash
            variant_kernels[v]<<<1, 1>>>(d_midstate, d_tail, nonce, d_hash_test);
            CUDA_CHECK(cudaDeviceSynchronize());

            uint32_t hash_ref[8], hash_test[8];
            CUDA_CHECK(cudaMemcpy(hash_ref, d_hash_ref, 32, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hash_test, d_hash_test, 32, cudaMemcpyDeviceToHost));

            for (int i = 0; i < 8; i++) {
                if (hash_ref[i] != hash_test[i]) {
                    printf("  FAIL: %s mismatch at iter %d, word %d\n", variant_names[v], iter, i);
                    variant_pass = false;
                    all_pass = false;
                    break;
                }
            }
        }

        if (variant_pass) {
            printf("  PASS: %s\n", variant_names[v]);
        }
    }

    cudaFree(d_header);
    cudaFree(d_midstate);
    cudaFree(d_tail);
    cudaFree(d_hash_ref);
    cudaFree(d_hash_test);

    return all_pass;
}

//==============================================================================
// Main Benchmark
//==============================================================================

int main(int argc, char** argv) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("\n");
    printf("================================================================================\n");
    printf("     TETSUO GPU Miner - Comprehensive Optimization Benchmark\n");
    printf("================================================================================\n");
    printf("\nDevice: %s\n", prop.name);
    printf("SMs: %d, Compute: %d.%d, Memory: %.1f GB\n",
           prop.multiProcessorCount, prop.major, prop.minor, prop.totalGlobalMem / 1e9);

    // Verify correctness first
    printf("\n");
    printf("================================================================================\n");
    printf("     CORRECTNESS VERIFICATION\n");
    printf("================================================================================\n\n");

    if (!verify_all_variants()) {
        printf("\nSome variants produce incorrect results. Aborting.\n");
        return 1;
    }

    printf("\nAll variants verified correct.\n");

    // Benchmark parameters
    int sm_count = prop.multiProcessorCount;
    int grid_size = sm_count * 256;
    int block_size = 256;
    int iterations = 200;

    printf("\n");
    printf("================================================================================\n");
    printf("     PERFORMANCE BENCHMARKS\n");
    printf("================================================================================\n");
    printf("\nConfig: grid=%d, block=%d, iterations=%d\n\n", grid_size, block_size, iterations);

    // Run benchmarks
    printf("Running benchmarks...\n\n");

    // Baseline (no midstate)
    g_baseline_rate = benchmark_baseline_kernel(grid_size, block_size, iterations);
    g_results.push_back({"Baseline (no midstate)", g_baseline_rate, 1.0});

    // Midstate only
    double rate = benchmark_kernel(launch_midstate, "Midstate", grid_size, block_size, iterations);
    g_results.push_back({"Midstate only", rate, rate / g_baseline_rate});

    // Individual optimizations
    rate = benchmark_kernel(launch_lop3, "LOP3", grid_size, block_size, iterations);
    g_results.push_back({"LOP3 instructions", rate, rate / g_baseline_rate});

    rate = benchmark_kernel(launch_early_exit, "Early Exit", grid_size, block_size, iterations);
    g_results.push_back({"Early MSW exit", rate, rate / g_baseline_rate});

    rate = benchmark_kernel(launch_sliding, "Sliding", grid_size, block_size, iterations);
    g_results.push_back({"Sliding window W[16]", rate, rate / g_baseline_rate});

    rate = benchmark_kernel(launch_shared, "Shared", grid_size, block_size, iterations);
    g_results.push_back({"Shared memory", rate, rate / g_baseline_rate});

    rate = benchmark_kernel(launch_precomp, "Precomp", grid_size, block_size, iterations);
    g_results.push_back({"SHA256-2 precompute", rate, rate / g_baseline_rate});

    // Combinations
    rate = benchmark_kernel(launch_lop3_sliding, "LOP3+Sliding", grid_size, block_size, iterations);
    g_results.push_back({"LOP3 + Sliding", rate, rate / g_baseline_rate});

    rate = benchmark_kernel(launch_lop3_shared, "LOP3+Shared", grid_size, block_size, iterations);
    g_results.push_back({"LOP3 + Shared", rate, rate / g_baseline_rate});

    rate = benchmark_kernel(launch_lop3_sliding_shared, "LOP3+Sliding+Shared", grid_size, block_size, iterations);
    g_results.push_back({"LOP3 + Sliding + Shared", rate, rate / g_baseline_rate});

    rate = benchmark_kernel(launch_all_optimizations, "All", grid_size, block_size, iterations);
    g_results.push_back({"ALL OPTIMIZATIONS", rate, rate / g_baseline_rate});

    // Print results table
    printf("\n");
    printf("================================================================================\n");
    printf("     RESULTS\n");
    printf("================================================================================\n\n");

    printf("+----------------------------------+------------+----------+\n");
    printf("| Optimization                     | Hash Rate  | Speedup  |\n");
    printf("+----------------------------------+------------+----------+\n");

    for (const auto& r : g_results) {
        printf("| %-32s | %7.2f MH | %6.2fx  |\n", r.name.c_str(), r.hash_rate_mh, r.speedup);
    }

    printf("+----------------------------------+------------+----------+\n");

    // Find best
    auto best = std::max_element(g_results.begin(), g_results.end(),
        [](const auto& a, const auto& b) { return a.hash_rate_mh < b.hash_rate_mh; });

    printf("\nBest: %s @ %.2f MH/s (%.2fx vs baseline)\n",
           best->name.c_str(), best->hash_rate_mh, best->speedup);

    // Analysis
    printf("\n");
    printf("================================================================================\n");
    printf("     ANALYSIS\n");
    printf("================================================================================\n\n");

    printf("Individual optimization impact vs baseline:\n\n");

    for (size_t i = 1; i < g_results.size() - 4; i++) {  // Skip baseline and combos
        double delta = (g_results[i].speedup - 1.0) * 100;
        printf("  %-25s: %+.1f%%\n", g_results[i].name.c_str(), delta);
    }

    printf("\nCombination synergies:\n\n");
    for (size_t i = g_results.size() - 4; i < g_results.size(); i++) {
        double delta = (g_results[i].speedup - 1.0) * 100;
        printf("  %-25s: %+.1f%%\n", g_results[i].name.c_str(), delta);
    }

    printf("\n");

    return 0;
}
