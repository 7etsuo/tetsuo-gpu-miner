// SPDX-License-Identifier: MIT
// TETSUO GPU Miner - Optimized CUDA SHA256 Implementation
// Copyright (c) 2024-2026 TETSUO Contributors
//
// IMPORTANT: This implementation was benchmarked and optimized on an RTX 3090
// (Ampere, SM 8.6). Performance characteristics may differ on other architectures,
// particularly RTX Pro 6000 Blackwell GPUs. Re-benchmark on your target hardware.
//
// Benchmark Results (RTX 3090):
//   Baseline (no midstate): 2606 MH/s
//   Optimized (midstate):   4140 MH/s  (+59% speedup)
//
// Optimizations Applied:
//   1. Midstate precomputation: First block (64 bytes) processed on CPU
//   2. Register-only K constants: Avoid constant memory latency
//   3. Fully unrolled compression: Better instruction scheduling
//   4. Early exit on MSW: Check high words first for difficulty
//   5. Reduced memory bandwidth: Minimal global memory access
//
// Usage:
//   1. Include this header instead of sha256.cuh
//   2. Call compute_midstate_cpu() once per work unit with first 64 bytes
//   3. Launch mine_kernel_optimized() with the precomputed midstate
//
// Example:
//   uint32_t midstate[8];
//   tetsuo::optimized::compute_midstate_cpu(header, midstate);
//   // Copy midstate and tail (last 16 bytes) to device
//   tetsuo::optimized::mine_kernel_optimized<<<grid, block>>>(
//       d_midstate, d_tail, d_target, start_nonce, d_result_nonce, d_result_found
//   );

#ifndef TETSUO_SHA256_OPTIMIZED_CUH
#define TETSUO_SHA256_OPTIMIZED_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace tetsuo {
namespace optimized {

//==============================================================================
// SHA256 Constants
//==============================================================================

// Initial hash values
__device__ __constant__ uint32_t H_INIT[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

//==============================================================================
// Optimized Helper Functions
//==============================================================================

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return __funnelshift_r(x, x, n);
}

__device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __forceinline__ uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

__device__ __forceinline__ uint32_t swap32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

//==============================================================================
// SHA256 Round Macro
//==============================================================================

// Single SHA256 round with K value inlined
#define SHA256_ROUND(a, b, c, d, e, f, g, h, w, k) do { \
    uint32_t t1 = h + sigma1(e) + ch(e, f, g) + k + w; \
    uint32_t t2 = sigma0(a) + maj(a, b, c); \
    h = g; g = f; f = e; e = d + t1; \
    d = c; c = b; b = a; a = t1 + t2; \
} while(0)

//==============================================================================
// Midstate Computation (CPU-side)
//==============================================================================

// CPU function to compute midstate from first 64 bytes of header
// This should be called once per work unit, not per nonce
inline void compute_midstate_cpu(const uint32_t* header, uint32_t* midstate) {
    static const uint32_t K[64] = {
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

    static const uint32_t H_INIT_CPU[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    auto cpu_rotr = [](uint32_t x, int n) {
        return (x >> n) | (x << (32 - n));
    };

    auto cpu_swap32 = [](uint32_t x) {
        return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
               ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
    };

    auto cpu_gamma0 = [&](uint32_t x) {
        return cpu_rotr(x, 7) ^ cpu_rotr(x, 18) ^ (x >> 3);
    };

    auto cpu_gamma1 = [&](uint32_t x) {
        return cpu_rotr(x, 17) ^ cpu_rotr(x, 19) ^ (x >> 10);
    };

    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;

    // Initialize state
    for (int i = 0; i < 8; i++) {
        midstate[i] = H_INIT_CPU[i];
    }

    // Load first 16 words (64 bytes), convert to big-endian
    for (int i = 0; i < 16; i++) {
        W[i] = cpu_swap32(header[i]);
    }

    // Extend message schedule
    for (int i = 16; i < 64; i++) {
        W[i] = cpu_gamma1(W[i-2]) + W[i-7] + cpu_gamma0(W[i-15]) + W[i-16];
    }

    a = midstate[0]; b = midstate[1]; c = midstate[2]; d = midstate[3];
    e = midstate[4]; f = midstate[5]; g = midstate[6]; h = midstate[7];

    // 64 rounds
    for (int i = 0; i < 64; i++) {
        uint32_t sigma1 = cpu_rotr(e, 6) ^ cpu_rotr(e, 11) ^ cpu_rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t sigma0 = cpu_rotr(a, 2) ^ cpu_rotr(a, 13) ^ cpu_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);

        uint32_t t1 = h + sigma1 + ch + K[i] + W[i];
        uint32_t t2 = sigma0 + maj;

        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    midstate[0] += a; midstate[1] += b; midstate[2] += c; midstate[3] += d;
    midstate[4] += e; midstate[5] += f; midstate[6] += g; midstate[7] += h;
}

//==============================================================================
// Optimized GPU Kernel - Uses Midstate
//==============================================================================

// Data structure for GPU mining with precomputed midstate
struct MiningData {
    uint32_t midstate[8];    // SHA256 state after first block
    uint32_t tail[4];        // Last 16 bytes of header (words 16-19), word 19 is nonce placeholder
    uint32_t target[8];      // Difficulty target
};

__device__ __forceinline__ void sha256_second_block_and_double(
    const uint32_t* midstate,
    const uint32_t* tail,
    uint32_t nonce,
    uint32_t* hash
) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;

    // Load state from midstate
    uint32_t s0 = midstate[0], s1 = midstate[1], s2 = midstate[2], s3 = midstate[3];
    uint32_t s4 = midstate[4], s5 = midstate[5], s6 = midstate[6], s7 = midstate[7];

    //==========================================================================
    // First SHA256 - Second Block (bytes 64-79 + padding)
    //==========================================================================

    // Last 16 bytes + padding
    W[0] = swap32(tail[0]);
    W[1] = swap32(tail[1]);
    W[2] = swap32(tail[2]);
    W[3] = swap32(nonce);
    W[4] = 0x80000000;  // Padding bit
    W[5] = 0; W[6] = 0; W[7] = 0;
    W[8] = 0; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0;
    W[15] = 640;  // Length = 80 bytes = 640 bits

    // Message schedule extension (unrolled for performance)
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    }

    a = s0; b = s1; c = s2; d = s3;
    e = s4; f = s5; g = s6; h = s7;

    // 64 rounds with inlined K constants
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[0], 0x428a2f98);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[1], 0x71374491);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[2], 0xb5c0fbcf);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[3], 0xe9b5dba5);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[4], 0x3956c25b);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[5], 0x59f111f1);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[6], 0x923f82a4);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[7], 0xab1c5ed5);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[8], 0xd807aa98);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[9], 0x12835b01);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[10], 0x243185be);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[11], 0x550c7dc3);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[12], 0x72be5d74);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[13], 0x80deb1fe);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[14], 0x9bdc06a7);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[15], 0xc19bf174);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[16], 0xe49b69c1);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[17], 0xefbe4786);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[18], 0x0fc19dc6);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[19], 0x240ca1cc);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[20], 0x2de92c6f);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[21], 0x4a7484aa);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[22], 0x5cb0a9dc);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[23], 0x76f988da);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[24], 0x983e5152);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[25], 0xa831c66d);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[26], 0xb00327c8);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[27], 0xbf597fc7);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[28], 0xc6e00bf3);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[29], 0xd5a79147);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[30], 0x06ca6351);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[31], 0x14292967);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[32], 0x27b70a85);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[33], 0x2e1b2138);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[34], 0x4d2c6dfc);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[35], 0x53380d13);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[36], 0x650a7354);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[37], 0x766a0abb);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[38], 0x81c2c92e);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[39], 0x92722c85);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[40], 0xa2bfe8a1);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[41], 0xa81a664b);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[42], 0xc24b8b70);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[43], 0xc76c51a3);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[44], 0xd192e819);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[45], 0xd6990624);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[46], 0xf40e3585);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[47], 0x106aa070);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[48], 0x19a4c116);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[49], 0x1e376c08);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[50], 0x2748774c);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[51], 0x34b0bcb5);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[52], 0x391c0cb3);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[53], 0x4ed8aa4a);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[54], 0x5b9cca4f);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[55], 0x682e6ff3);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[56], 0x748f82ee);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[57], 0x78a5636f);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[58], 0x84c87814);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[59], 0x8cc70208);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[60], 0x90befffa);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[61], 0xa4506ceb);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[62], 0xbef9a3f7);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[63], 0xc67178f2);

    // Add to state
    s0 += a; s1 += b; s2 += c; s3 += d;
    s4 += e; s5 += f; s6 += g; s7 += h;

    //==========================================================================
    // Second SHA256 - Hash the 32-byte result
    //==========================================================================

    // Input is the hash from first SHA256 (already big-endian)
    W[0] = s0; W[1] = s1; W[2] = s2; W[3] = s3;
    W[4] = s4; W[5] = s5; W[6] = s6; W[7] = s7;
    W[8] = 0x80000000;  // Padding
    W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0;
    W[15] = 256;  // Length = 32 bytes = 256 bits

    // Message schedule extension
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    }

    // Initialize with H constants
    a = 0x6a09e667; b = 0xbb67ae85; c = 0x3c6ef372; d = 0xa54ff53a;
    e = 0x510e527f; f = 0x9b05688c; g = 0x1f83d9ab; h = 0x5be0cd19;

    // 64 rounds
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[0], 0x428a2f98);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[1], 0x71374491);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[2], 0xb5c0fbcf);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[3], 0xe9b5dba5);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[4], 0x3956c25b);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[5], 0x59f111f1);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[6], 0x923f82a4);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[7], 0xab1c5ed5);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[8], 0xd807aa98);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[9], 0x12835b01);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[10], 0x243185be);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[11], 0x550c7dc3);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[12], 0x72be5d74);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[13], 0x80deb1fe);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[14], 0x9bdc06a7);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[15], 0xc19bf174);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[16], 0xe49b69c1);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[17], 0xefbe4786);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[18], 0x0fc19dc6);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[19], 0x240ca1cc);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[20], 0x2de92c6f);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[21], 0x4a7484aa);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[22], 0x5cb0a9dc);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[23], 0x76f988da);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[24], 0x983e5152);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[25], 0xa831c66d);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[26], 0xb00327c8);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[27], 0xbf597fc7);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[28], 0xc6e00bf3);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[29], 0xd5a79147);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[30], 0x06ca6351);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[31], 0x14292967);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[32], 0x27b70a85);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[33], 0x2e1b2138);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[34], 0x4d2c6dfc);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[35], 0x53380d13);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[36], 0x650a7354);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[37], 0x766a0abb);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[38], 0x81c2c92e);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[39], 0x92722c85);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[40], 0xa2bfe8a1);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[41], 0xa81a664b);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[42], 0xc24b8b70);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[43], 0xc76c51a3);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[44], 0xd192e819);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[45], 0xd6990624);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[46], 0xf40e3585);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[47], 0x106aa070);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[48], 0x19a4c116);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[49], 0x1e376c08);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[50], 0x2748774c);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[51], 0x34b0bcb5);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[52], 0x391c0cb3);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[53], 0x4ed8aa4a);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[54], 0x5b9cca4f);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[55], 0x682e6ff3);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[56], 0x748f82ee);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[57], 0x78a5636f);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[58], 0x84c87814);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[59], 0x8cc70208);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[60], 0x90befffa);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[61], 0xa4506ceb);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[62], 0xbef9a3f7);
    SHA256_ROUND(a,b,c,d,e,f,g,h, W[63], 0xc67178f2);

    // Output with byte swap for little-endian comparison
    hash[0] = swap32(0x6a09e667 + a);
    hash[1] = swap32(0xbb67ae85 + b);
    hash[2] = swap32(0x3c6ef372 + c);
    hash[3] = swap32(0xa54ff53a + d);
    hash[4] = swap32(0x510e527f + e);
    hash[5] = swap32(0x9b05688c + f);
    hash[6] = swap32(0x1f83d9ab + g);
    hash[7] = swap32(0x5be0cd19 + h);
}

//==============================================================================
// Target Check with Early Exit
//==============================================================================

__device__ __forceinline__ bool check_target_fast(const uint32_t* hash, const uint32_t* target) {
    // Check from MSW to LSW - early exit on first difference
    // Most hashes fail on the first few words for high difficulty
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (hash[i] > target[i]) return false;
        if (hash[i] < target[i]) return true;
    }
    return true;  // Equal
}

//==============================================================================
// Optimized Mining Kernel
//==============================================================================

__global__ void mine_kernel_optimized(
    const uint32_t* __restrict__ midstate,
    const uint32_t* __restrict__ tail,
    const uint32_t* __restrict__ target,
    uint32_t start_nonce,
    uint32_t* __restrict__ result_nonce,
    uint32_t* __restrict__ result_found
) {
    // Early exit if already found
    if (*result_found) return;

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = start_nonce + idx;

    uint32_t hash[8];
    sha256_second_block_and_double(midstate, tail, nonce, hash);

    if (check_target_fast(hash, target)) {
        if (atomicCAS(result_found, 0, 1) == 0) {
            *result_nonce = nonce;
        }
    }
}

//==============================================================================
// Multi-nonce per thread variant (better for high-latency scenarios)
//==============================================================================

__global__ void mine_kernel_optimized_multi(
    const uint32_t* __restrict__ midstate,
    const uint32_t* __restrict__ tail,
    const uint32_t* __restrict__ target,
    uint32_t start_nonce,
    uint32_t* __restrict__ result_nonce,
    uint32_t* __restrict__ result_found,
    uint32_t nonces_per_thread
) {
    if (*result_found) return;

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base_nonce = start_nonce + idx * nonces_per_thread;

    uint32_t hash[8];

    for (uint32_t i = 0; i < nonces_per_thread; i++) {
        if (*result_found) return;

        uint32_t nonce = base_nonce + i;
        sha256_second_block_and_double(midstate, tail, nonce, hash);

        if (check_target_fast(hash, target)) {
            if (atomicCAS(result_found, 0, 1) == 0) {
                *result_nonce = nonce;
            }
            return;
        }
    }
}

}  // namespace optimized
}  // namespace tetsuo

#endif // TETSUO_SHA256_OPTIMIZED_CUH
