// SPDX-License-Identifier: MIT
// TETSUO GPU Miner - Performance Benchmark Suite
// Copyright (c) 2024-2026 TETSUO Contributors
//
// Compares baseline vs optimized kernel performance

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

#include "sha256.cuh"
#include "sha256_optimized.cuh"
#include "tetsuo/cuda_utils.cuh"

using namespace tetsuo;

//==============================================================================
// Benchmark Infrastructure
//==============================================================================

struct BenchmarkResult {
    const char* name;
    double hash_rate_mh;
    double kernel_time_ms;
    uint64_t total_hashes;
    int grid_size;
    int block_size;
};

void print_result(const BenchmarkResult& r) {
    printf("  %-30s %8.2f MH/s  (grid=%d, block=%d)\n",
           r.name, r.hash_rate_mh, r.grid_size, r.block_size);
}

//==============================================================================
// Baseline Kernel Benchmark
//==============================================================================

BenchmarkResult benchmark_baseline(
    int grid_size,
    int block_size,
    int iterations
) {
    uint32_t header[20];
    for (int i = 0; i < 20; i++) {
        header[i] = 0x12345678 + i;
    }

    // Impossible target
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

    // Warmup
    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_result_found, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
    mine_kernel<<<grid_size, block_size>>>(d_header, d_target, 0, d_result_nonce, d_result_found);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
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

    return {"baseline", hash_rate, elapsed_ms, total_hashes, grid_size, block_size};
}

//==============================================================================
// Optimized Kernel Benchmark
//==============================================================================

BenchmarkResult benchmark_optimized(
    int grid_size,
    int block_size,
    int iterations
) {
    uint32_t header[20];
    for (int i = 0; i < 20; i++) {
        header[i] = 0x12345678 + i;
    }

    // Compute midstate on CPU
    uint32_t midstate[8];
    optimized::compute_midstate_cpu(header, midstate);

    // Tail is last 4 words (bytes 64-79)
    uint32_t tail[4] = {header[16], header[17], header[18], header[19]};

    // Impossible target
    uint32_t target[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    uint32_t* d_midstate;
    uint32_t* d_tail;
    uint32_t* d_target;
    uint32_t* d_result_nonce;
    uint32_t* d_result_found;

    CUDA_CHECK(cudaMalloc(&d_midstate, 32));
    CUDA_CHECK(cudaMalloc(&d_tail, 16));
    CUDA_CHECK(cudaMalloc(&d_target, 32));
    CUDA_CHECK(cudaMalloc(&d_result_nonce, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result_found, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_midstate, midstate, 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tail, tail, 16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, target, 32, cudaMemcpyHostToDevice));

    // Warmup
    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_result_found, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
    optimized::mine_kernel_optimized<<<grid_size, block_size>>>(
        d_midstate, d_tail, d_target, 0, d_result_nonce, d_result_found);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    uint32_t nonce = 0;
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemcpy(d_result_found, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
        optimized::mine_kernel_optimized<<<grid_size, block_size>>>(
            d_midstate, d_tail, d_target, nonce, d_result_nonce, d_result_found);
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
    cudaFree(d_tail);
    cudaFree(d_target);
    cudaFree(d_result_nonce);
    cudaFree(d_result_found);

    return {"optimized (midstate)", hash_rate, elapsed_ms, total_hashes, grid_size, block_size};
}

//==============================================================================
// Optimized Multi-nonce Kernel Benchmark
//==============================================================================

BenchmarkResult benchmark_optimized_multi(
    int grid_size,
    int block_size,
    int nonces_per_thread,
    int iterations
) {
    uint32_t header[20];
    for (int i = 0; i < 20; i++) {
        header[i] = 0x12345678 + i;
    }

    uint32_t midstate[8];
    optimized::compute_midstate_cpu(header, midstate);

    uint32_t tail[4] = {header[16], header[17], header[18], header[19]};
    uint32_t target[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    uint32_t* d_midstate;
    uint32_t* d_tail;
    uint32_t* d_target;
    uint32_t* d_result_nonce;
    uint32_t* d_result_found;

    CUDA_CHECK(cudaMalloc(&d_midstate, 32));
    CUDA_CHECK(cudaMalloc(&d_tail, 16));
    CUDA_CHECK(cudaMalloc(&d_target, 32));
    CUDA_CHECK(cudaMalloc(&d_result_nonce, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result_found, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_midstate, midstate, 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tail, tail, 16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, target, 32, cudaMemcpyHostToDevice));

    // Warmup
    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_result_found, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
    optimized::mine_kernel_optimized_multi<<<grid_size, block_size>>>(
        d_midstate, d_tail, d_target, 0, d_result_nonce, d_result_found, nonces_per_thread);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    uint32_t nonce = 0;
    uint32_t hashes_per_launch = grid_size * block_size * nonces_per_thread;
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemcpy(d_result_found, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
        optimized::mine_kernel_optimized_multi<<<grid_size, block_size>>>(
            d_midstate, d_tail, d_target, nonce, d_result_nonce, d_result_found, nonces_per_thread);
        nonce += hashes_per_launch;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    uint64_t total_hashes = (uint64_t)hashes_per_launch * iterations;
    double hash_rate = (total_hashes / (elapsed_ms / 1000.0)) / 1e6;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_midstate);
    cudaFree(d_tail);
    cudaFree(d_target);
    cudaFree(d_result_nonce);
    cudaFree(d_result_found);

    char name[64];
    snprintf(name, sizeof(name), "optimized multi (n=%d)", nonces_per_thread);
    return {strdup(name), hash_rate, elapsed_ms, total_hashes, grid_size, block_size};
}

//==============================================================================
// Correctness Verification - GPU Kernels
//==============================================================================

__global__ void compute_hash_baseline(const uint32_t* header, uint32_t nonce, uint32_t* hash) {
    sha256_block_header(header, nonce, hash);
}

__global__ void compute_hash_optimized(const uint32_t* midstate, const uint32_t* tail, uint32_t nonce, uint32_t* hash) {
    optimized::sha256_second_block_and_double(midstate, tail, nonce, hash);
}

bool verify_hash_match() {
    printf("\n[VERIFY] Direct hash comparison (baseline vs optimized)...\n");

    srand(12345);

    uint32_t header[20];
    for (int i = 0; i < 20; i++) {
        header[i] = rand();
    }

    uint32_t midstate[8];
    optimized::compute_midstate_cpu(header, midstate);
    uint32_t tail[4] = {header[16], header[17], header[18], header[19]};

    uint32_t* d_header;
    uint32_t* d_midstate;
    uint32_t* d_tail;
    uint32_t* d_hash1;
    uint32_t* d_hash2;

    CUDA_CHECK(cudaMalloc(&d_header, 80));
    CUDA_CHECK(cudaMalloc(&d_midstate, 32));
    CUDA_CHECK(cudaMalloc(&d_tail, 16));
    CUDA_CHECK(cudaMalloc(&d_hash1, 32));
    CUDA_CHECK(cudaMalloc(&d_hash2, 32));

    CUDA_CHECK(cudaMemcpy(d_header, header, 80, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_midstate, midstate, 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tail, tail, 16, cudaMemcpyHostToDevice));

    bool all_match = true;
    int num_tests = 1000;

    for (int iter = 0; iter < num_tests && all_match; iter++) {
        uint32_t nonce = rand();

        compute_hash_baseline<<<1, 1>>>(d_header, nonce, d_hash1);
        compute_hash_optimized<<<1, 1>>>(d_midstate, d_tail, nonce, d_hash2);
        CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t hash1[8], hash2[8];
        CUDA_CHECK(cudaMemcpy(hash1, d_hash1, 32, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hash2, d_hash2, 32, cudaMemcpyDeviceToHost));

        for (int i = 0; i < 8; i++) {
            if (hash1[i] != hash2[i]) {
                printf("  FAIL: Mismatch at iter %d, nonce %u, word %d\n", iter, nonce, i);
                printf("    Baseline:  %08x, Optimized: %08x\n", hash1[i], hash2[i]);
                all_match = false;
                break;
            }
        }
    }

    cudaFree(d_header);
    cudaFree(d_midstate);
    cudaFree(d_tail);
    cudaFree(d_hash1);
    cudaFree(d_hash2);

    if (all_match) {
        printf("  PASS: %d hashes match between baseline and optimized\n", num_tests);
    }

    return all_match;
}

bool verify_mining_finds_solution() {
    printf("\n[VERIFY] Mining kernel finds correct solutions...\n");

    srand(54321);

    uint32_t header[20];
    for (int i = 0; i < 20; i++) {
        header[i] = rand();
    }

    uint32_t midstate[8];
    optimized::compute_midstate_cpu(header, midstate);
    uint32_t tail[4] = {header[16], header[17], header[18], header[19]};

    // Easy target - about 1 in 256 chance (MSW high byte must be 0)
    uint32_t target[8] = {
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00FFFFFF
    };

    uint32_t* d_header;
    uint32_t* d_midstate;
    uint32_t* d_tail;
    uint32_t* d_target;
    uint32_t* d_result_nonce;
    uint32_t* d_result_found;

    CUDA_CHECK(cudaMalloc(&d_header, 80));
    CUDA_CHECK(cudaMalloc(&d_midstate, 32));
    CUDA_CHECK(cudaMalloc(&d_tail, 16));
    CUDA_CHECK(cudaMalloc(&d_target, 32));
    CUDA_CHECK(cudaMalloc(&d_result_nonce, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result_found, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_header, header, 80, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_midstate, midstate, 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tail, tail, 16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, target, 32, cudaMemcpyHostToDevice));

    // Test baseline
    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_result_found, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));

    int grid_size = 256;
    int block_size = 256;
    uint32_t nonce = 0;
    uint32_t baseline_nonce = 0;
    bool baseline_found = false;

    for (int batch = 0; batch < 1000 && !baseline_found; batch++) {
        mine_kernel<<<grid_size, block_size>>>(d_header, d_target, nonce, d_result_nonce, d_result_found);
        CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t found;
        CUDA_CHECK(cudaMemcpy(&found, d_result_found, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (found) {
            CUDA_CHECK(cudaMemcpy(&baseline_nonce, d_result_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            baseline_found = true;
        }
        nonce += grid_size * block_size;
    }

    if (!baseline_found) {
        printf("  FAIL: Baseline kernel did not find solution\n");
        cudaFree(d_header);
        cudaFree(d_midstate);
        cudaFree(d_tail);
        cudaFree(d_target);
        cudaFree(d_result_nonce);
        cudaFree(d_result_found);
        return false;
    }

    printf("  Baseline found nonce: %u\n", baseline_nonce);

    // Verify baseline solution
    uint32_t* d_hash;
    CUDA_CHECK(cudaMalloc(&d_hash, 32));
    compute_hash_baseline<<<1, 1>>>(d_header, baseline_nonce, d_hash);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t hash[8];
    CUDA_CHECK(cudaMemcpy(hash, d_hash, 32, cudaMemcpyDeviceToHost));

    // Check target
    bool valid = true;
    for (int i = 7; i >= 0; i--) {
        if (hash[i] > target[i]) { valid = false; break; }
        if (hash[i] < target[i]) break;
    }

    if (!valid) {
        printf("  FAIL: Baseline solution does not meet target\n");
        printf("  Hash:   ");
        for (int i = 7; i >= 0; i--) printf("%08x", hash[i]);
        printf("\n  Target: ");
        for (int i = 7; i >= 0; i--) printf("%08x", target[i]);
        printf("\n");
        cudaFree(d_hash);
        cudaFree(d_header);
        cudaFree(d_midstate);
        cudaFree(d_tail);
        cudaFree(d_target);
        cudaFree(d_result_nonce);
        cudaFree(d_result_found);
        return false;
    }

    printf("  Baseline solution verified\n");

    // Now verify optimized kernel finds a solution too
    CUDA_CHECK(cudaMemcpy(d_result_found, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));

    nonce = 0;
    uint32_t optimized_nonce = 0;
    bool optimized_found = false;

    for (int batch = 0; batch < 1000 && !optimized_found; batch++) {
        optimized::mine_kernel_optimized<<<grid_size, block_size>>>(
            d_midstate, d_tail, d_target, nonce, d_result_nonce, d_result_found);
        CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t found;
        CUDA_CHECK(cudaMemcpy(&found, d_result_found, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (found) {
            CUDA_CHECK(cudaMemcpy(&optimized_nonce, d_result_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            optimized_found = true;
        }
        nonce += grid_size * block_size;
    }

    if (!optimized_found) {
        printf("  FAIL: Optimized kernel did not find solution\n");
        cudaFree(d_hash);
        cudaFree(d_header);
        cudaFree(d_midstate);
        cudaFree(d_tail);
        cudaFree(d_target);
        cudaFree(d_result_nonce);
        cudaFree(d_result_found);
        return false;
    }

    printf("  Optimized found nonce: %u\n", optimized_nonce);

    // Verify optimized solution using baseline hash function
    compute_hash_baseline<<<1, 1>>>(d_header, optimized_nonce, d_hash);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hash, d_hash, 32, cudaMemcpyDeviceToHost));

    valid = true;
    for (int i = 7; i >= 0; i--) {
        if (hash[i] > target[i]) { valid = false; break; }
        if (hash[i] < target[i]) break;
    }

    cudaFree(d_hash);
    cudaFree(d_header);
    cudaFree(d_midstate);
    cudaFree(d_tail);
    cudaFree(d_target);
    cudaFree(d_result_nonce);
    cudaFree(d_result_found);

    if (!valid) {
        printf("  FAIL: Optimized solution does not meet target\n");
        return false;
    }

    printf("  PASS: Both kernels find valid solutions\n");
    return true;
}

//==============================================================================
// Main
//==============================================================================

int main(int argc, char** argv) {
    srand(42);

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
    printf("========================================================\n");
    printf("     TETSUO GPU Miner - Benchmark Suite\n");
    printf("========================================================\n");
    printf("\nDevice: %s\n", prop.name);
    printf("SMs: %d, Compute: %d.%d\n", prop.multiProcessorCount, prop.major, prop.minor);
    printf("Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);

    // Correctness verification first
    printf("\n========================================================\n");
    printf("     CORRECTNESS VERIFICATION\n");
    printf("========================================================\n");

    bool hash_correct = verify_hash_match();
    if (!hash_correct) {
        printf("\nOptimized kernel produces incorrect hashes! Aborting.\n");
        return 1;
    }

    bool mining_correct = verify_mining_finds_solution();
    if (!mining_correct) {
        printf("\nMining kernel verification failed! Aborting.\n");
        return 1;
    }

    // Performance benchmarks
    printf("\n========================================================\n");
    printf("     PERFORMANCE BENCHMARKS\n");
    printf("========================================================\n");

    int sm_count = prop.multiProcessorCount;
    int iterations = 200;

    // Find optimal configuration for each kernel
    std::vector<BenchmarkResult> baseline_results;
    std::vector<BenchmarkResult> optimized_results;

    struct Config { int grid_mult; int block_size; };
    Config configs[] = {
        {128, 256},
        {256, 256},
        {512, 256},
        {256, 128},
        {256, 512},
        {1024, 256},
    };

    printf("\n--- Baseline Kernel ---\n");
    for (const auto& cfg : configs) {
        int grid = sm_count * cfg.grid_mult;
        auto result = benchmark_baseline(grid, cfg.block_size, iterations);
        baseline_results.push_back(result);
        print_result(result);
    }

    printf("\n--- Optimized Kernel (Midstate) ---\n");
    for (const auto& cfg : configs) {
        int grid = sm_count * cfg.grid_mult;
        auto result = benchmark_optimized(grid, cfg.block_size, iterations);
        optimized_results.push_back(result);
        print_result(result);
    }

    printf("\n--- Optimized Multi-Nonce Kernel ---\n");
    int multi_nonces[] = {2, 4, 8};
    std::vector<BenchmarkResult> multi_results;
    for (int n : multi_nonces) {
        int grid = sm_count * 256;
        auto result = benchmark_optimized_multi(grid, 256, n, iterations);
        multi_results.push_back(result);
        print_result(result);
    }

    // Find best results
    auto best_baseline = *std::max_element(baseline_results.begin(), baseline_results.end(),
        [](const auto& a, const auto& b) { return a.hash_rate_mh < b.hash_rate_mh; });

    auto best_optimized = *std::max_element(optimized_results.begin(), optimized_results.end(),
        [](const auto& a, const auto& b) { return a.hash_rate_mh < b.hash_rate_mh; });

    auto best_multi = *std::max_element(multi_results.begin(), multi_results.end(),
        [](const auto& a, const auto& b) { return a.hash_rate_mh < b.hash_rate_mh; });

    printf("\n========================================================\n");
    printf("     SUMMARY\n");
    printf("========================================================\n");

    printf("\nBest Results:\n");
    printf("  Baseline:      %8.2f MH/s\n", best_baseline.hash_rate_mh);
    printf("  Optimized:     %8.2f MH/s\n", best_optimized.hash_rate_mh);
    printf("  Multi-nonce:   %8.2f MH/s\n", best_multi.hash_rate_mh);

    double speedup_opt = best_optimized.hash_rate_mh / best_baseline.hash_rate_mh;
    double speedup_multi = best_multi.hash_rate_mh / best_baseline.hash_rate_mh;

    printf("\nSpeedup vs Baseline:\n");
    printf("  Optimized:     %.2fx\n", speedup_opt);
    printf("  Multi-nonce:   %.2fx\n", speedup_multi);

    double best_overall = std::max({best_baseline.hash_rate_mh, best_optimized.hash_rate_mh, best_multi.hash_rate_mh});
    printf("\nBest Overall: %.2f MH/s (%.2fx vs baseline)\n",
           best_overall, best_overall / best_baseline.hash_rate_mh);

    printf("\n");

    return 0;
}
