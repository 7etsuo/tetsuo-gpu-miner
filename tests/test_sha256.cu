// SPDX-License-Identifier: MIT
// TETSUO GPU Miner - SHA256 Correctness and Performance Tests
// Copyright (c) 2024-2026 TETSUO Contributors

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>

#include "sha256.cuh"
#include "tetsuo/cuda_utils.cuh"

using namespace tetsuo;

//==============================================================================
// Test Utilities
//==============================================================================

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        g_tests_failed++; \
        return false; \
    } \
} while(0)

#define TEST_PASS(name) do { \
    printf("  PASS: %s\n", name); \
    g_tests_passed++; \
    return true; \
} while(0)

void print_hex(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        printf("%02x", data[i]);
    }
}

void print_hex32(const uint32_t* data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        printf("%08x", data[i]);
    }
}

std::string hex_string(const uint32_t* data, size_t count) {
    std::ostringstream ss;
    for (size_t i = 0; i < count; i++) {
        ss << std::hex << std::setfill('0') << std::setw(8) << data[i];
    }
    return ss.str();
}

void hex_to_bytes(const char* hex, uint8_t* out, size_t len) {
    for (size_t i = 0; i < len; i++) {
        unsigned int byte;
        sscanf(hex + 2*i, "%02x", &byte);
        out[i] = (uint8_t)byte;
    }
}

//==============================================================================
// CPU Reference Implementation
//==============================================================================

static const uint32_t CPU_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static const uint32_t CPU_H_INIT[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

inline uint32_t cpu_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

inline uint32_t cpu_ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

inline uint32_t cpu_maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

inline uint32_t cpu_sigma0(uint32_t x) {
    return cpu_rotr(x, 2) ^ cpu_rotr(x, 13) ^ cpu_rotr(x, 22);
}

inline uint32_t cpu_sigma1(uint32_t x) {
    return cpu_rotr(x, 6) ^ cpu_rotr(x, 11) ^ cpu_rotr(x, 25);
}

inline uint32_t cpu_gamma0(uint32_t x) {
    return cpu_rotr(x, 7) ^ cpu_rotr(x, 18) ^ (x >> 3);
}

inline uint32_t cpu_gamma1(uint32_t x) {
    return cpu_rotr(x, 17) ^ cpu_rotr(x, 19) ^ (x >> 10);
}

inline uint32_t cpu_swap32(uint32_t x) {
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
           ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
}

// CPU reference SHA256d for block header
void cpu_sha256_block_header(const uint32_t* header, uint32_t nonce, uint32_t* hash) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t t1, t2;

    // First SHA256
    uint32_t state[8];
    for (int i = 0; i < 8; i++) {
        state[i] = CPU_H_INIT[i];
    }

    // Block 1: bytes 0-63
    for (int i = 0; i < 16; i++) {
        W[i] = cpu_swap32(header[i]);
    }

    for (int i = 16; i < 64; i++) {
        W[i] = cpu_gamma1(W[i-2]) + W[i-7] + cpu_gamma0(W[i-15]) + W[i-16];
    }

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    for (int i = 0; i < 64; i++) {
        t1 = h + cpu_sigma1(e) + cpu_ch(e, f, g) + CPU_K[i] + W[i];
        t2 = cpu_sigma0(a) + cpu_maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;

    // Block 2: bytes 64-79 + padding
    W[0] = cpu_swap32(header[16]);
    W[1] = cpu_swap32(header[17]);
    W[2] = cpu_swap32(header[18]);
    W[3] = cpu_swap32(nonce);
    W[4] = 0x80000000;
    for (int i = 5; i < 15; i++) W[i] = 0;
    W[15] = 640;

    for (int i = 16; i < 64; i++) {
        W[i] = cpu_gamma1(W[i-2]) + W[i-7] + cpu_gamma0(W[i-15]) + W[i-16];
    }

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    for (int i = 0; i < 64; i++) {
        t1 = h + cpu_sigma1(e) + cpu_ch(e, f, g) + CPU_K[i] + W[i];
        t2 = cpu_sigma0(a) + cpu_maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;

    // Second SHA256
    uint32_t state2[8];
    for (int i = 0; i < 8; i++) {
        state2[i] = CPU_H_INIT[i];
    }

    W[0] = state[0]; W[1] = state[1]; W[2] = state[2]; W[3] = state[3];
    W[4] = state[4]; W[5] = state[5]; W[6] = state[6]; W[7] = state[7];
    W[8] = 0x80000000;
    for (int i = 9; i < 15; i++) W[i] = 0;
    W[15] = 256;

    for (int i = 16; i < 64; i++) {
        W[i] = cpu_gamma1(W[i-2]) + W[i-7] + cpu_gamma0(W[i-15]) + W[i-16];
    }

    a = state2[0]; b = state2[1]; c = state2[2]; d = state2[3];
    e = state2[4]; f = state2[5]; g = state2[6]; h = state2[7];

    for (int i = 0; i < 64; i++) {
        t1 = h + cpu_sigma1(e) + cpu_ch(e, f, g) + CPU_K[i] + W[i];
        t2 = cpu_sigma0(a) + cpu_maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    hash[0] = cpu_swap32(state2[0] + a);
    hash[1] = cpu_swap32(state2[1] + b);
    hash[2] = cpu_swap32(state2[2] + c);
    hash[3] = cpu_swap32(state2[3] + d);
    hash[4] = cpu_swap32(state2[4] + e);
    hash[5] = cpu_swap32(state2[5] + f);
    hash[6] = cpu_swap32(state2[6] + g);
    hash[7] = cpu_swap32(state2[7] + h);
}

bool cpu_check_target(const uint32_t* hash, const uint32_t* target) {
    for (int i = 7; i >= 0; i--) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return true;
}

//==============================================================================
// GPU Test Kernels
//==============================================================================

__global__ void test_sha256_kernel(
    const uint32_t* header,
    uint32_t nonce,
    uint32_t* hash_out
) {
    sha256_block_header(header, nonce, hash_out);
}

__global__ void test_check_target_kernel(
    const uint32_t* hash,
    const uint32_t* target,
    int* result
) {
    *result = check_target(hash, target) ? 1 : 0;
}

//==============================================================================
// Correctness Tests
//==============================================================================

bool test_sha256_known_vector() {
    printf("\n[TEST] SHA256 Known Test Vector (Bitcoin Block #1)\n");

    // Bitcoin Block #1 header (80 bytes) - a well-known test vector
    // Block 1 header from Bitcoin mainnet
    uint8_t header_bytes[80] = {
        0x01, 0x00, 0x00, 0x00, // version
        // prev block hash (genesis)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // merkle root
        0x3b, 0xa3, 0xed, 0xfd, 0x7a, 0x7b, 0x12, 0xb2,
        0x7a, 0xc7, 0x2c, 0x3e, 0x67, 0x76, 0x8f, 0x61,
        0x7f, 0xc8, 0x1b, 0xc3, 0x88, 0x8a, 0x51, 0x32,
        0x3a, 0x9f, 0xb8, 0xaa, 0x4b, 0x1e, 0x5e, 0x4a,
        // timestamp
        0x29, 0xab, 0x5f, 0x49,
        // bits
        0xff, 0xff, 0x00, 0x1d,
        // nonce
        0x1d, 0xac, 0x2b, 0x7c
    };

    uint32_t header[20];
    memcpy(header, header_bytes, 80);
    uint32_t nonce = 0x7c2bac1d;  // Block 1's actual nonce (little-endian in header)

    // Compute on CPU
    uint32_t cpu_hash[8];
    cpu_sha256_block_header(header, nonce, cpu_hash);

    printf("  CPU hash: ");
    print_hex32(cpu_hash, 8);
    printf("\n");

    // Compute on GPU
    uint32_t* d_header;
    uint32_t* d_hash;
    CUDA_CHECK(cudaMalloc(&d_header, 80));
    CUDA_CHECK(cudaMalloc(&d_hash, 32));
    CUDA_CHECK(cudaMemcpy(d_header, header, 80, cudaMemcpyHostToDevice));

    test_sha256_kernel<<<1, 1>>>(d_header, nonce, d_hash);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t gpu_hash[8];
    CUDA_CHECK(cudaMemcpy(gpu_hash, d_hash, 32, cudaMemcpyDeviceToHost));

    printf("  GPU hash: ");
    print_hex32(gpu_hash, 8);
    printf("\n");

    cudaFree(d_header);
    cudaFree(d_hash);

    // Compare
    bool match = true;
    for (int i = 0; i < 8; i++) {
        if (cpu_hash[i] != gpu_hash[i]) {
            match = false;
            break;
        }
    }

    TEST_ASSERT(match, "CPU and GPU hashes must match");
    TEST_PASS("SHA256 known vector");
}

bool test_sha256_random_nonces() {
    printf("\n[TEST] SHA256 Random Nonces (1000 iterations)\n");

    // Create random header
    uint32_t header[20];
    for (int i = 0; i < 20; i++) {
        header[i] = rand();
    }

    uint32_t* d_header;
    uint32_t* d_hash;
    CUDA_CHECK(cudaMalloc(&d_header, 80));
    CUDA_CHECK(cudaMalloc(&d_hash, 32));
    CUDA_CHECK(cudaMemcpy(d_header, header, 80, cudaMemcpyHostToDevice));

    for (int iter = 0; iter < 1000; iter++) {
        uint32_t nonce = rand();

        // CPU
        uint32_t cpu_hash[8];
        cpu_sha256_block_header(header, nonce, cpu_hash);

        // GPU
        test_sha256_kernel<<<1, 1>>>(d_header, nonce, d_hash);
        CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t gpu_hash[8];
        CUDA_CHECK(cudaMemcpy(gpu_hash, d_hash, 32, cudaMemcpyDeviceToHost));

        for (int i = 0; i < 8; i++) {
            if (cpu_hash[i] != gpu_hash[i]) {
                printf("  Mismatch at nonce %u, word %d\n", nonce, i);
                printf("  CPU: %08x, GPU: %08x\n", cpu_hash[i], gpu_hash[i]);
                cudaFree(d_header);
                cudaFree(d_hash);
                TEST_ASSERT(false, "Hash mismatch");
            }
        }
    }

    cudaFree(d_header);
    cudaFree(d_hash);

    TEST_PASS("SHA256 random nonces (1000 iterations)");
}

bool test_target_comparison() {
    printf("\n[TEST] Target Comparison\n");

    // Test cases: hash vs target
    struct TestCase {
        uint32_t hash[8];
        uint32_t target[8];
        bool expected;
        const char* desc;
    };

    TestCase cases[] = {
        // Hash less than target (valid)
        {{0, 0, 0, 0, 0, 0, 0, 0x00000001},
         {0, 0, 0, 0, 0, 0, 0, 0x00000002},
         true, "hash < target (MSW)"},

        // Hash greater than target (invalid)
        {{0, 0, 0, 0, 0, 0, 0, 0x00000003},
         {0, 0, 0, 0, 0, 0, 0, 0x00000002},
         false, "hash > target (MSW)"},

        // Hash equal to target (valid)
        {{1, 2, 3, 4, 5, 6, 7, 8},
         {1, 2, 3, 4, 5, 6, 7, 8},
         true, "hash == target"},

        // Hash less in LSW only
        {{0x00000001, 0, 0, 0, 0, 0, 0, 0x00000001},
         {0x00000002, 0, 0, 0, 0, 0, 0, 0x00000001},
         true, "hash < target (LSW, MSW equal)"},

        // Typical mining scenario: many leading zeros
        {{0x12345678, 0, 0, 0, 0, 0, 0, 0},
         {0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, 0, 0, 0},
         true, "typical mining (leading zeros)"},
    };

    int num_cases = sizeof(cases) / sizeof(cases[0]);

    uint32_t* d_hash;
    uint32_t* d_target;
    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_hash, 32));
    CUDA_CHECK(cudaMalloc(&d_target, 32));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    for (int i = 0; i < num_cases; i++) {
        // CPU check
        bool cpu_result = cpu_check_target(cases[i].hash, cases[i].target);

        // GPU check
        CUDA_CHECK(cudaMemcpy(d_hash, cases[i].hash, 32, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_target, cases[i].target, 32, cudaMemcpyHostToDevice));

        test_check_target_kernel<<<1, 1>>>(d_hash, d_target, d_result);
        CUDA_CHECK(cudaDeviceSynchronize());

        int gpu_result;
        CUDA_CHECK(cudaMemcpy(&gpu_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

        printf("  Case %d (%s): CPU=%d, GPU=%d, expected=%d\n",
               i, cases[i].desc, cpu_result, gpu_result, cases[i].expected);

        if (cpu_result != cases[i].expected) {
            cudaFree(d_hash);
            cudaFree(d_target);
            cudaFree(d_result);
            TEST_ASSERT(false, "CPU result mismatch");
        }

        if ((gpu_result != 0) != cases[i].expected) {
            cudaFree(d_hash);
            cudaFree(d_target);
            cudaFree(d_result);
            TEST_ASSERT(false, "GPU result mismatch");
        }
    }

    cudaFree(d_hash);
    cudaFree(d_target);
    cudaFree(d_result);

    TEST_PASS("Target comparison");
}

bool test_mining_kernel_finds_solution() {
    printf("\n[TEST] Mining Kernel Finds Known Solution\n");

    // Create a header and find a nonce that produces a hash with leading zeros
    uint32_t header[20];
    for (int i = 0; i < 20; i++) {
        header[i] = 0x12345678 + i;
    }

    // Set an easy target (about 1 in 256 chance - MSW byte 7 must be 0)
    uint32_t target[8] = {
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00FFFFFF
    };

    // Find a solution on CPU first
    uint32_t winning_nonce = 0;
    bool found = false;
    for (uint32_t n = 0; n < 1000000 && !found; n++) {
        uint32_t hash[8];
        cpu_sha256_block_header(header, n, hash);
        if (cpu_check_target(hash, target)) {
            winning_nonce = n;
            found = true;
        }
    }

    TEST_ASSERT(found, "CPU must find solution for easy target");
    printf("  CPU found nonce: %u\n", winning_nonce);

    // Now test GPU kernel
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

    // Launch with enough threads to find the solution
    int grid_size = (winning_nonce / 256) + 1;
    mine_kernel<<<grid_size, 256>>>(d_header, d_target, 0, d_result_nonce, d_result_found);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t result_found, result_nonce;
    CUDA_CHECK(cudaMemcpy(&result_found, d_result_found, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&result_nonce, d_result_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    printf("  GPU found: %s, nonce: %u\n", result_found ? "yes" : "no", result_nonce);

    cudaFree(d_header);
    cudaFree(d_target);
    cudaFree(d_result_nonce);
    cudaFree(d_result_found);

    TEST_ASSERT(result_found != 0, "GPU must find solution");

    // Verify the GPU's solution is valid
    uint32_t verify_hash[8];
    cpu_sha256_block_header(header, result_nonce, verify_hash);
    TEST_ASSERT(cpu_check_target(verify_hash, target), "GPU solution must be valid");

    TEST_PASS("Mining kernel finds solution");
}

//==============================================================================
// Performance Benchmarks
//==============================================================================

struct BenchmarkResult {
    double hash_rate_mh;
    double kernel_time_ms;
    uint64_t total_hashes;
};

BenchmarkResult benchmark_kernel(
    void (*launch_fn)(const uint32_t*, const uint32_t*, uint32_t, uint32_t*, uint32_t*, int, int),
    const char* name,
    int grid_size,
    int block_size,
    int iterations
) {
    printf("\n[BENCHMARK] %s\n", name);
    printf("  Grid: %d, Block: %d, Iterations: %d\n", grid_size, block_size, iterations);

    uint32_t header[20];
    for (int i = 0; i < 20; i++) {
        header[i] = rand();
    }

    // Impossible target - no solution will be found
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
    launch_fn(d_header, d_target, 0, d_result_nonce, d_result_found, grid_size, block_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    uint32_t nonce = 0;
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemcpy(d_result_found, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
        launch_fn(d_header, d_target, nonce, d_result_nonce, d_result_found, grid_size, block_size);
        nonce += grid_size * block_size;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    uint64_t total_hashes = (uint64_t)grid_size * block_size * iterations;
    double hash_rate = (total_hashes / (elapsed_ms / 1000.0)) / 1e6;

    printf("  Total hashes: %lu\n", total_hashes);
    printf("  Time: %.2f ms\n", elapsed_ms);
    printf("  Hash rate: %.2f MH/s\n", hash_rate);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_header);
    cudaFree(d_target);
    cudaFree(d_result_nonce);
    cudaFree(d_result_found);

    return {hash_rate, elapsed_ms, total_hashes};
}

void launch_baseline_kernel(
    const uint32_t* d_header,
    const uint32_t* d_target,
    uint32_t start_nonce,
    uint32_t* d_result_nonce,
    uint32_t* d_result_found,
    int grid_size,
    int block_size
) {
    mine_kernel<<<grid_size, block_size>>>(
        d_header, d_target, start_nonce, d_result_nonce, d_result_found
    );
}

//==============================================================================
// Main
//==============================================================================

void run_correctness_tests() {
    printf("\n");
    printf("========================================\n");
    printf("       CORRECTNESS TESTS\n");
    printf("========================================\n");

    test_sha256_known_vector();
    test_sha256_random_nonces();
    test_target_comparison();
    test_mining_kernel_finds_solution();

    printf("\n----------------------------------------\n");
    printf("Results: %d passed, %d failed\n", g_tests_passed, g_tests_failed);
    printf("----------------------------------------\n");
}

void run_performance_benchmarks() {
    printf("\n");
    printf("========================================\n");
    printf("       PERFORMANCE BENCHMARKS\n");
    printf("========================================\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\nDevice: %s\n", prop.name);
    printf("SMs: %d, Max threads/block: %d\n",
           prop.multiProcessorCount, prop.maxThreadsPerBlock);

    // Test different configurations
    int sm_count = prop.multiProcessorCount;

    struct Config {
        int grid_mult;
        int block_size;
    };

    Config configs[] = {
        {128, 256},
        {256, 256},
        {512, 256},
        {256, 128},
        {256, 512},
    };

    printf("\n--- Baseline Kernel ---\n");

    double best_rate = 0;
    int best_grid = 0, best_block = 0;

    for (const auto& cfg : configs) {
        int grid = sm_count * cfg.grid_mult;
        auto result = benchmark_kernel(
            launch_baseline_kernel,
            "mine_kernel",
            grid,
            cfg.block_size,
            100
        );

        if (result.hash_rate_mh > best_rate) {
            best_rate = result.hash_rate_mh;
            best_grid = grid;
            best_block = cfg.block_size;
        }
    }

    printf("\n----------------------------------------\n");
    printf("Best baseline config: grid=%d, block=%d\n", best_grid, best_block);
    printf("Best hash rate: %.2f MH/s\n", best_rate);
    printf("----------------------------------------\n");
}

int main(int argc, char** argv) {
    srand(42);  // Deterministic for reproducibility

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));

    bool run_correctness = true;
    bool run_perf = true;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--correctness") == 0) {
            run_perf = false;
        } else if (strcmp(argv[i], "--perf") == 0) {
            run_correctness = false;
        }
    }

    if (run_correctness) {
        run_correctness_tests();
    }

    if (run_perf) {
        run_performance_benchmarks();
    }

    if (g_tests_failed > 0) {
        return 1;
    }

    return 0;
}
