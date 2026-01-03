// SPDX-License-Identifier: MIT
// TETSUO GPU Miner - SHA256 Optimization Variants
// Copyright (c) 2024-2026 TETSUO Contributors
//
// IMPORTANT: These variants were benchmarked on an RTX 3090 (Ampere, SM 8.6).
// Performance characteristics may differ significantly on other architectures,
// particularly RTX Pro 6000 Blackwell GPUs. Re-benchmark on your target hardware.
//
// Contains multiple optimization strategies for benchmarking:
//   1. LOP3 ternary logic instructions
//   2. Early MSW exit
//   3. W sliding window (16-element circular buffer)
//   4. Shared memory for block-constant data
//   5. Second SHA256 precomputation
//   6. Combinations of the above

#ifndef TETSUO_SHA256_VARIANTS_CUH
#define TETSUO_SHA256_VARIANTS_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace tetsuo {
namespace variants {

//==============================================================================
// Common Helper Functions
//==============================================================================

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return __funnelshift_r(x, x, n);
}

__device__ __forceinline__ uint32_t swap32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

// Standard implementations
__device__ __forceinline__ uint32_t ch_std(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t maj_std(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

// LOP3 implementations - single instruction for 3-input boolean
__device__ __forceinline__ uint32_t ch_lop3(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(r) : "r"(x), "r"(y), "r"(z));
    return r;
}

__device__ __forceinline__ uint32_t maj_lop3(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(r) : "r"(x), "r"(y), "r"(z));
    return r;
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

// K constants in registers
__device__ __constant__ uint32_t K[64] = {
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

//==============================================================================
// CPU Midstate Computation (shared by all variants)
//==============================================================================

inline void compute_midstate_cpu(const uint32_t* header, uint32_t* midstate) {
    static const uint32_t K_CPU[64] = {
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

    static const uint32_t H_INIT[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    auto cpu_rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    auto cpu_swap32 = [](uint32_t x) {
        return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
               ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
    };
    auto cpu_gamma0 = [&](uint32_t x) { return cpu_rotr(x, 7) ^ cpu_rotr(x, 18) ^ (x >> 3); };
    auto cpu_gamma1 = [&](uint32_t x) { return cpu_rotr(x, 17) ^ cpu_rotr(x, 19) ^ (x >> 10); };

    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;

    for (int i = 0; i < 8; i++) midstate[i] = H_INIT[i];
    for (int i = 0; i < 16; i++) W[i] = cpu_swap32(header[i]);
    for (int i = 16; i < 64; i++) W[i] = cpu_gamma1(W[i-2]) + W[i-7] + cpu_gamma0(W[i-15]) + W[i-16];

    a = midstate[0]; b = midstate[1]; c = midstate[2]; d = midstate[3];
    e = midstate[4]; f = midstate[5]; g = midstate[6]; h = midstate[7];

    for (int i = 0; i < 64; i++) {
        uint32_t S1 = cpu_rotr(e, 6) ^ cpu_rotr(e, 11) ^ cpu_rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t S0 = cpu_rotr(a, 2) ^ cpu_rotr(a, 13) ^ cpu_rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t1 = h + S1 + ch + K_CPU[i] + W[i];
        uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    midstate[0] += a; midstate[1] += b; midstate[2] += c; midstate[3] += d;
    midstate[4] += e; midstate[5] += f; midstate[6] += g; midstate[7] += h;
}

//==============================================================================
// VARIANT 1: LOP3 Instructions Only
//==============================================================================

#define ROUND_LOP3(a,b,c,d,e,f,g,h,w,k) do { \
    uint32_t t1 = h + sigma1(e) + ch_lop3(e,f,g) + k + w; \
    uint32_t t2 = sigma0(a) + maj_lop3(a,b,c); \
    h = g; g = f; f = e; e = d + t1; \
    d = c; c = b; b = a; a = t1 + t2; \
} while(0)

__device__ void sha256_lop3(
    const uint32_t* midstate,
    const uint32_t* tail,
    uint32_t nonce,
    uint32_t* hash
) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;

    uint32_t s0 = midstate[0], s1 = midstate[1], s2 = midstate[2], s3 = midstate[3];
    uint32_t s4 = midstate[4], s5 = midstate[5], s6 = midstate[6], s7 = midstate[7];

    // First SHA256 - Second block
    W[0] = swap32(tail[0]); W[1] = swap32(tail[1]); W[2] = swap32(tail[2]); W[3] = swap32(nonce);
    W[4] = 0x80000000; W[5] = 0; W[6] = 0; W[7] = 0;
    W[8] = 0; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 640;

    #pragma unroll
    for (int i = 16; i < 64; i++) W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];

    a = s0; b = s1; c = s2; d = s3; e = s4; f = s5; g = s6; h = s7;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        ROUND_LOP3(a,b,c,d,e,f,g,h, W[i], K[i]);
    }

    s0 += a; s1 += b; s2 += c; s3 += d; s4 += e; s5 += f; s6 += g; s7 += h;

    // Second SHA256
    W[0] = s0; W[1] = s1; W[2] = s2; W[3] = s3;
    W[4] = s4; W[5] = s5; W[6] = s6; W[7] = s7;
    W[8] = 0x80000000; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 256;

    #pragma unroll
    for (int i = 16; i < 64; i++) W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];

    a = 0x6a09e667; b = 0xbb67ae85; c = 0x3c6ef372; d = 0xa54ff53a;
    e = 0x510e527f; f = 0x9b05688c; g = 0x1f83d9ab; h = 0x5be0cd19;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        ROUND_LOP3(a,b,c,d,e,f,g,h, W[i], K[i]);
    }

    hash[0] = swap32(0x6a09e667 + a); hash[1] = swap32(0xbb67ae85 + b);
    hash[2] = swap32(0x3c6ef372 + c); hash[3] = swap32(0xa54ff53a + d);
    hash[4] = swap32(0x510e527f + e); hash[5] = swap32(0x9b05688c + f);
    hash[6] = swap32(0x1f83d9ab + g); hash[7] = swap32(0x5be0cd19 + h);
}

__global__ void mine_lop3(
    const uint32_t* __restrict__ midstate,
    const uint32_t* __restrict__ tail,
    const uint32_t* __restrict__ target,
    uint32_t start_nonce,
    uint32_t* __restrict__ result_nonce,
    uint32_t* __restrict__ result_found
) {
    if (*result_found) return;
    uint32_t nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t hash[8];
    sha256_lop3(midstate, tail, nonce, hash);

    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (hash[i] > target[i]) return;
        if (hash[i] < target[i]) break;
    }
    if (atomicCAS(result_found, 0, 1) == 0) *result_nonce = nonce;
}

//==============================================================================
// VARIANT 2: Early MSW Exit
//==============================================================================

__device__ bool sha256_early_exit(
    const uint32_t* midstate,
    const uint32_t* tail,
    uint32_t nonce,
    const uint32_t* target,
    uint32_t* hash
) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;

    uint32_t s0 = midstate[0], s1 = midstate[1], s2 = midstate[2], s3 = midstate[3];
    uint32_t s4 = midstate[4], s5 = midstate[5], s6 = midstate[6], s7 = midstate[7];

    // First SHA256 - Second block
    W[0] = swap32(tail[0]); W[1] = swap32(tail[1]); W[2] = swap32(tail[2]); W[3] = swap32(nonce);
    W[4] = 0x80000000; W[5] = 0; W[6] = 0; W[7] = 0;
    W[8] = 0; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 640;

    #pragma unroll
    for (int i = 16; i < 64; i++) W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];

    a = s0; b = s1; c = s2; d = s3; e = s4; f = s5; g = s6; h = s7;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch_std(e,f,g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj_std(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    s0 += a; s1 += b; s2 += c; s3 += d; s4 += e; s5 += f; s6 += g; s7 += h;

    // Second SHA256
    W[0] = s0; W[1] = s1; W[2] = s2; W[3] = s3;
    W[4] = s4; W[5] = s5; W[6] = s6; W[7] = s7;
    W[8] = 0x80000000; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 256;

    #pragma unroll
    for (int i = 16; i < 64; i++) W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];

    a = 0x6a09e667; b = 0xbb67ae85; c = 0x3c6ef372; d = 0xa54ff53a;
    e = 0x510e527f; f = 0x9b05688c; g = 0x1f83d9ab; h = 0x5be0cd19;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch_std(e,f,g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj_std(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    // Early exit check on MSW (hash[7])
    uint32_t h7 = swap32(0x5be0cd19 + h);
    if (h7 > target[7]) return false;

    uint32_t h6 = swap32(0x1f83d9ab + g);
    if (h7 == target[7] && h6 > target[6]) return false;

    // Compute full hash only if MSW passes
    hash[7] = h7;
    hash[6] = h6;
    hash[5] = swap32(0x9b05688c + f);
    hash[4] = swap32(0x510e527f + e);
    hash[3] = swap32(0xa54ff53a + d);
    hash[2] = swap32(0x3c6ef372 + c);
    hash[1] = swap32(0xbb67ae85 + b);
    hash[0] = swap32(0x6a09e667 + a);

    return true;
}

__global__ void mine_early_exit(
    const uint32_t* __restrict__ midstate,
    const uint32_t* __restrict__ tail,
    const uint32_t* __restrict__ target,
    uint32_t start_nonce,
    uint32_t* __restrict__ result_nonce,
    uint32_t* __restrict__ result_found
) {
    if (*result_found) return;
    uint32_t nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t hash[8];

    if (!sha256_early_exit(midstate, tail, nonce, target, hash)) return;

    // Full target check
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (hash[i] > target[i]) return;
        if (hash[i] < target[i]) break;
    }
    if (atomicCAS(result_found, 0, 1) == 0) *result_nonce = nonce;
}

//==============================================================================
// VARIANT 3: Sliding Window W (16-element circular buffer)
//==============================================================================

__device__ void sha256_sliding(
    const uint32_t* midstate,
    const uint32_t* tail,
    uint32_t nonce,
    uint32_t* hash
) {
    uint32_t W[16];  // Only 16 elements instead of 64
    uint32_t a, b, c, d, e, f, g, h;

    uint32_t s0 = midstate[0], s1 = midstate[1], s2 = midstate[2], s3 = midstate[3];
    uint32_t s4 = midstate[4], s5 = midstate[5], s6 = midstate[6], s7 = midstate[7];

    // First SHA256 - Second block
    W[0] = swap32(tail[0]); W[1] = swap32(tail[1]); W[2] = swap32(tail[2]); W[3] = swap32(nonce);
    W[4] = 0x80000000; W[5] = 0; W[6] = 0; W[7] = 0;
    W[8] = 0; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 640;

    a = s0; b = s1; c = s2; d = s3; e = s4; f = s5; g = s6; h = s7;

    // Rounds 0-15: use preloaded W
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t t1 = h + sigma1(e) + ch_std(e,f,g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj_std(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    // Rounds 16-63: compute W on the fly with sliding window
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t w_new = gamma1(W[(i-2) & 15]) + W[(i-7) & 15] + gamma0(W[(i-15) & 15]) + W[i & 15];
        W[i & 15] = w_new;
        uint32_t t1 = h + sigma1(e) + ch_std(e,f,g) + K[i] + w_new;
        uint32_t t2 = sigma0(a) + maj_std(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    s0 += a; s1 += b; s2 += c; s3 += d; s4 += e; s5 += f; s6 += g; s7 += h;

    // Second SHA256
    W[0] = s0; W[1] = s1; W[2] = s2; W[3] = s3;
    W[4] = s4; W[5] = s5; W[6] = s6; W[7] = s7;
    W[8] = 0x80000000; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 256;

    a = 0x6a09e667; b = 0xbb67ae85; c = 0x3c6ef372; d = 0xa54ff53a;
    e = 0x510e527f; f = 0x9b05688c; g = 0x1f83d9ab; h = 0x5be0cd19;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t t1 = h + sigma1(e) + ch_std(e,f,g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj_std(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t w_new = gamma1(W[(i-2) & 15]) + W[(i-7) & 15] + gamma0(W[(i-15) & 15]) + W[i & 15];
        W[i & 15] = w_new;
        uint32_t t1 = h + sigma1(e) + ch_std(e,f,g) + K[i] + w_new;
        uint32_t t2 = sigma0(a) + maj_std(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    hash[0] = swap32(0x6a09e667 + a); hash[1] = swap32(0xbb67ae85 + b);
    hash[2] = swap32(0x3c6ef372 + c); hash[3] = swap32(0xa54ff53a + d);
    hash[4] = swap32(0x510e527f + e); hash[5] = swap32(0x9b05688c + f);
    hash[6] = swap32(0x1f83d9ab + g); hash[7] = swap32(0x5be0cd19 + h);
}

__global__ void mine_sliding(
    const uint32_t* __restrict__ midstate,
    const uint32_t* __restrict__ tail,
    const uint32_t* __restrict__ target,
    uint32_t start_nonce,
    uint32_t* __restrict__ result_nonce,
    uint32_t* __restrict__ result_found
) {
    if (*result_found) return;
    uint32_t nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t hash[8];
    sha256_sliding(midstate, tail, nonce, hash);

    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (hash[i] > target[i]) return;
        if (hash[i] < target[i]) break;
    }
    if (atomicCAS(result_found, 0, 1) == 0) *result_nonce = nonce;
}

//==============================================================================
// VARIANT 4: Shared Memory for Block-Constant Data
//==============================================================================

__global__ void mine_shared(
    const uint32_t* __restrict__ midstate,
    const uint32_t* __restrict__ tail,
    const uint32_t* __restrict__ target,
    uint32_t start_nonce,
    uint32_t* __restrict__ result_nonce,
    uint32_t* __restrict__ result_found
) {
    __shared__ uint32_t s_midstate[8];
    __shared__ uint32_t s_tail[4];
    __shared__ uint32_t s_target[8];

    // Cooperative load into shared memory
    if (threadIdx.x < 8) {
        s_midstate[threadIdx.x] = midstate[threadIdx.x];
        s_target[threadIdx.x] = target[threadIdx.x];
    }
    if (threadIdx.x < 4) {
        s_tail[threadIdx.x] = tail[threadIdx.x];
    }
    __syncthreads();

    if (*result_found) return;

    uint32_t nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;

    uint32_t s0 = s_midstate[0], s1 = s_midstate[1], s2 = s_midstate[2], s3 = s_midstate[3];
    uint32_t s4 = s_midstate[4], s5 = s_midstate[5], s6 = s_midstate[6], s7 = s_midstate[7];

    W[0] = swap32(s_tail[0]); W[1] = swap32(s_tail[1]); W[2] = swap32(s_tail[2]); W[3] = swap32(nonce);
    W[4] = 0x80000000; W[5] = 0; W[6] = 0; W[7] = 0;
    W[8] = 0; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 640;

    #pragma unroll
    for (int i = 16; i < 64; i++) W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];

    a = s0; b = s1; c = s2; d = s3; e = s4; f = s5; g = s6; h = s7;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch_std(e,f,g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj_std(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    s0 += a; s1 += b; s2 += c; s3 += d; s4 += e; s5 += f; s6 += g; s7 += h;

    W[0] = s0; W[1] = s1; W[2] = s2; W[3] = s3;
    W[4] = s4; W[5] = s5; W[6] = s6; W[7] = s7;
    W[8] = 0x80000000; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 256;

    #pragma unroll
    for (int i = 16; i < 64; i++) W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];

    a = 0x6a09e667; b = 0xbb67ae85; c = 0x3c6ef372; d = 0xa54ff53a;
    e = 0x510e527f; f = 0x9b05688c; g = 0x1f83d9ab; h = 0x5be0cd19;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch_std(e,f,g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj_std(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    uint32_t hash[8];
    hash[0] = swap32(0x6a09e667 + a); hash[1] = swap32(0xbb67ae85 + b);
    hash[2] = swap32(0x3c6ef372 + c); hash[3] = swap32(0xa54ff53a + d);
    hash[4] = swap32(0x510e527f + e); hash[5] = swap32(0x9b05688c + f);
    hash[6] = swap32(0x1f83d9ab + g); hash[7] = swap32(0x5be0cd19 + h);

    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (hash[i] > s_target[i]) return;
        if (hash[i] < s_target[i]) break;
    }
    if (atomicCAS(result_found, 0, 1) == 0) *result_nonce = nonce;
}

//==============================================================================
// VARIANT 5: Second SHA256 Precomputation
// Precompute W[16..63] terms that don't depend on variable input
//==============================================================================

// For second SHA256, input is: s0-s7, 0x80000000, 0,0,0,0,0,0, 256
// W[8]=0x80000000, W[9..14]=0, W[15]=256 are constants
// We can precompute partial W values

__device__ void sha256_precomp(
    const uint32_t* midstate,
    const uint32_t* tail,
    uint32_t nonce,
    uint32_t* hash
) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;

    uint32_t s0 = midstate[0], s1 = midstate[1], s2 = midstate[2], s3 = midstate[3];
    uint32_t s4 = midstate[4], s5 = midstate[5], s6 = midstate[6], s7 = midstate[7];

    // First SHA256 - Second block (standard)
    W[0] = swap32(tail[0]); W[1] = swap32(tail[1]); W[2] = swap32(tail[2]); W[3] = swap32(nonce);
    W[4] = 0x80000000; W[5] = 0; W[6] = 0; W[7] = 0;
    W[8] = 0; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 640;

    #pragma unroll
    for (int i = 16; i < 64; i++) W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];

    a = s0; b = s1; c = s2; d = s3; e = s4; f = s5; g = s6; h = s7;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch_std(e,f,g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj_std(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    s0 += a; s1 += b; s2 += c; s3 += d; s4 += e; s5 += f; s6 += g; s7 += h;

    // Second SHA256 with partial precomputation
    // W[0..7] = s0..s7 (variable)
    // W[8] = 0x80000000, W[9..14] = 0, W[15] = 256 (constant)

    W[0] = s0; W[1] = s1; W[2] = s2; W[3] = s3;
    W[4] = s4; W[5] = s5; W[6] = s6; W[7] = s7;

    // Constants for second hash
    const uint32_t W8 = 0x80000000;
    const uint32_t W15 = 256;

    // W[16] = gamma1(W[14]) + W[9] + gamma0(W[1]) + W[0]
    //       = gamma1(0) + 0 + gamma0(s1) + s0 = gamma0(s1) + s0
    W[8] = W8;
    W[9] = 0; W[10] = 0; W[11] = 0; W[12] = 0; W[13] = 0; W[14] = 0;
    W[15] = W15;

    // Optimized W computation for known constants
    W[16] = gamma0(W[1]) + W[0];  // gamma1(0) = 0, W[9] = 0
    W[17] = gamma1(W[15]) + gamma0(W[2]) + W[1];  // W[10] = 0
    W[18] = gamma1(W[16]) + gamma0(W[3]) + W[2];  // W[11] = 0
    W[19] = gamma1(W[17]) + gamma0(W[4]) + W[3];  // W[12] = 0
    W[20] = gamma1(W[18]) + gamma0(W[5]) + W[4];  // W[13] = 0
    W[21] = gamma1(W[19]) + gamma0(W[6]) + W[5];  // W[14] = 0
    W[22] = gamma1(W[20]) + W[15] + gamma0(W[7]) + W[6];
    W[23] = gamma1(W[21]) + W[16] + gamma0(W[8]) + W[7];

    #pragma unroll
    for (int i = 24; i < 64; i++) {
        W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    }

    a = 0x6a09e667; b = 0xbb67ae85; c = 0x3c6ef372; d = 0xa54ff53a;
    e = 0x510e527f; f = 0x9b05688c; g = 0x1f83d9ab; h = 0x5be0cd19;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch_std(e,f,g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj_std(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    hash[0] = swap32(0x6a09e667 + a); hash[1] = swap32(0xbb67ae85 + b);
    hash[2] = swap32(0x3c6ef372 + c); hash[3] = swap32(0xa54ff53a + d);
    hash[4] = swap32(0x510e527f + e); hash[5] = swap32(0x9b05688c + f);
    hash[6] = swap32(0x1f83d9ab + g); hash[7] = swap32(0x5be0cd19 + h);
}

__global__ void mine_precomp(
    const uint32_t* __restrict__ midstate,
    const uint32_t* __restrict__ tail,
    const uint32_t* __restrict__ target,
    uint32_t start_nonce,
    uint32_t* __restrict__ result_nonce,
    uint32_t* __restrict__ result_found
) {
    if (*result_found) return;
    uint32_t nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t hash[8];
    sha256_precomp(midstate, tail, nonce, hash);

    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (hash[i] > target[i]) return;
        if (hash[i] < target[i]) break;
    }
    if (atomicCAS(result_found, 0, 1) == 0) *result_nonce = nonce;
}

//==============================================================================
// VARIANT 6: LOP3 + Sliding Window
//==============================================================================

__device__ void sha256_lop3_sliding(
    const uint32_t* midstate,
    const uint32_t* tail,
    uint32_t nonce,
    uint32_t* hash
) {
    uint32_t W[16];
    uint32_t a, b, c, d, e, f, g, h;

    uint32_t s0 = midstate[0], s1 = midstate[1], s2 = midstate[2], s3 = midstate[3];
    uint32_t s4 = midstate[4], s5 = midstate[5], s6 = midstate[6], s7 = midstate[7];

    W[0] = swap32(tail[0]); W[1] = swap32(tail[1]); W[2] = swap32(tail[2]); W[3] = swap32(nonce);
    W[4] = 0x80000000; W[5] = 0; W[6] = 0; W[7] = 0;
    W[8] = 0; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 640;

    a = s0; b = s1; c = s2; d = s3; e = s4; f = s5; g = s6; h = s7;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t t1 = h + sigma1(e) + ch_lop3(e,f,g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj_lop3(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t w_new = gamma1(W[(i-2) & 15]) + W[(i-7) & 15] + gamma0(W[(i-15) & 15]) + W[i & 15];
        W[i & 15] = w_new;
        uint32_t t1 = h + sigma1(e) + ch_lop3(e,f,g) + K[i] + w_new;
        uint32_t t2 = sigma0(a) + maj_lop3(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    s0 += a; s1 += b; s2 += c; s3 += d; s4 += e; s5 += f; s6 += g; s7 += h;

    W[0] = s0; W[1] = s1; W[2] = s2; W[3] = s3;
    W[4] = s4; W[5] = s5; W[6] = s6; W[7] = s7;
    W[8] = 0x80000000; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 256;

    a = 0x6a09e667; b = 0xbb67ae85; c = 0x3c6ef372; d = 0xa54ff53a;
    e = 0x510e527f; f = 0x9b05688c; g = 0x1f83d9ab; h = 0x5be0cd19;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t t1 = h + sigma1(e) + ch_lop3(e,f,g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj_lop3(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t w_new = gamma1(W[(i-2) & 15]) + W[(i-7) & 15] + gamma0(W[(i-15) & 15]) + W[i & 15];
        W[i & 15] = w_new;
        uint32_t t1 = h + sigma1(e) + ch_lop3(e,f,g) + K[i] + w_new;
        uint32_t t2 = sigma0(a) + maj_lop3(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    hash[0] = swap32(0x6a09e667 + a); hash[1] = swap32(0xbb67ae85 + b);
    hash[2] = swap32(0x3c6ef372 + c); hash[3] = swap32(0xa54ff53a + d);
    hash[4] = swap32(0x510e527f + e); hash[5] = swap32(0x9b05688c + f);
    hash[6] = swap32(0x1f83d9ab + g); hash[7] = swap32(0x5be0cd19 + h);
}

__global__ void mine_lop3_sliding(
    const uint32_t* __restrict__ midstate,
    const uint32_t* __restrict__ tail,
    const uint32_t* __restrict__ target,
    uint32_t start_nonce,
    uint32_t* __restrict__ result_nonce,
    uint32_t* __restrict__ result_found
) {
    if (*result_found) return;
    uint32_t nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t hash[8];
    sha256_lop3_sliding(midstate, tail, nonce, hash);

    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (hash[i] > target[i]) return;
        if (hash[i] < target[i]) break;
    }
    if (atomicCAS(result_found, 0, 1) == 0) *result_nonce = nonce;
}

//==============================================================================
// VARIANT 7: LOP3 + Shared Memory
//==============================================================================

__global__ void mine_lop3_shared(
    const uint32_t* __restrict__ midstate,
    const uint32_t* __restrict__ tail,
    const uint32_t* __restrict__ target,
    uint32_t start_nonce,
    uint32_t* __restrict__ result_nonce,
    uint32_t* __restrict__ result_found
) {
    __shared__ uint32_t s_midstate[8];
    __shared__ uint32_t s_tail[4];
    __shared__ uint32_t s_target[8];

    if (threadIdx.x < 8) {
        s_midstate[threadIdx.x] = midstate[threadIdx.x];
        s_target[threadIdx.x] = target[threadIdx.x];
    }
    if (threadIdx.x < 4) s_tail[threadIdx.x] = tail[threadIdx.x];
    __syncthreads();

    if (*result_found) return;

    uint32_t nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;

    uint32_t s0 = s_midstate[0], s1 = s_midstate[1], s2 = s_midstate[2], s3 = s_midstate[3];
    uint32_t s4 = s_midstate[4], s5 = s_midstate[5], s6 = s_midstate[6], s7 = s_midstate[7];

    W[0] = swap32(s_tail[0]); W[1] = swap32(s_tail[1]); W[2] = swap32(s_tail[2]); W[3] = swap32(nonce);
    W[4] = 0x80000000; W[5] = 0; W[6] = 0; W[7] = 0;
    W[8] = 0; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 640;

    #pragma unroll
    for (int i = 16; i < 64; i++) W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];

    a = s0; b = s1; c = s2; d = s3; e = s4; f = s5; g = s6; h = s7;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        ROUND_LOP3(a,b,c,d,e,f,g,h, W[i], K[i]);
    }

    s0 += a; s1 += b; s2 += c; s3 += d; s4 += e; s5 += f; s6 += g; s7 += h;

    W[0] = s0; W[1] = s1; W[2] = s2; W[3] = s3;
    W[4] = s4; W[5] = s5; W[6] = s6; W[7] = s7;
    W[8] = 0x80000000; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 256;

    #pragma unroll
    for (int i = 16; i < 64; i++) W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];

    a = 0x6a09e667; b = 0xbb67ae85; c = 0x3c6ef372; d = 0xa54ff53a;
    e = 0x510e527f; f = 0x9b05688c; g = 0x1f83d9ab; h = 0x5be0cd19;

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        ROUND_LOP3(a,b,c,d,e,f,g,h, W[i], K[i]);
    }

    uint32_t hash[8];
    hash[0] = swap32(0x6a09e667 + a); hash[1] = swap32(0xbb67ae85 + b);
    hash[2] = swap32(0x3c6ef372 + c); hash[3] = swap32(0xa54ff53a + d);
    hash[4] = swap32(0x510e527f + e); hash[5] = swap32(0x9b05688c + f);
    hash[6] = swap32(0x1f83d9ab + g); hash[7] = swap32(0x5be0cd19 + h);

    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (hash[i] > s_target[i]) return;
        if (hash[i] < s_target[i]) break;
    }
    if (atomicCAS(result_found, 0, 1) == 0) *result_nonce = nonce;
}

//==============================================================================
// VARIANT 8: LOP3 + Sliding + Shared (Triple Combo)
//==============================================================================

__global__ void mine_lop3_sliding_shared(
    const uint32_t* __restrict__ midstate,
    const uint32_t* __restrict__ tail,
    const uint32_t* __restrict__ target,
    uint32_t start_nonce,
    uint32_t* __restrict__ result_nonce,
    uint32_t* __restrict__ result_found
) {
    __shared__ uint32_t s_midstate[8];
    __shared__ uint32_t s_tail[4];
    __shared__ uint32_t s_target[8];

    if (threadIdx.x < 8) {
        s_midstate[threadIdx.x] = midstate[threadIdx.x];
        s_target[threadIdx.x] = target[threadIdx.x];
    }
    if (threadIdx.x < 4) s_tail[threadIdx.x] = tail[threadIdx.x];
    __syncthreads();

    if (*result_found) return;

    uint32_t nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t W[16];
    uint32_t a, b, c, d, e, f, g, h;

    uint32_t s0 = s_midstate[0], s1 = s_midstate[1], s2 = s_midstate[2], s3 = s_midstate[3];
    uint32_t s4 = s_midstate[4], s5 = s_midstate[5], s6 = s_midstate[6], s7 = s_midstate[7];

    W[0] = swap32(s_tail[0]); W[1] = swap32(s_tail[1]); W[2] = swap32(s_tail[2]); W[3] = swap32(nonce);
    W[4] = 0x80000000; W[5] = 0; W[6] = 0; W[7] = 0;
    W[8] = 0; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 640;

    a = s0; b = s1; c = s2; d = s3; e = s4; f = s5; g = s6; h = s7;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t t1 = h + sigma1(e) + ch_lop3(e,f,g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj_lop3(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t w_new = gamma1(W[(i-2) & 15]) + W[(i-7) & 15] + gamma0(W[(i-15) & 15]) + W[i & 15];
        W[i & 15] = w_new;
        uint32_t t1 = h + sigma1(e) + ch_lop3(e,f,g) + K[i] + w_new;
        uint32_t t2 = sigma0(a) + maj_lop3(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    s0 += a; s1 += b; s2 += c; s3 += d; s4 += e; s5 += f; s6 += g; s7 += h;

    W[0] = s0; W[1] = s1; W[2] = s2; W[3] = s3;
    W[4] = s4; W[5] = s5; W[6] = s6; W[7] = s7;
    W[8] = 0x80000000; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 256;

    a = 0x6a09e667; b = 0xbb67ae85; c = 0x3c6ef372; d = 0xa54ff53a;
    e = 0x510e527f; f = 0x9b05688c; g = 0x1f83d9ab; h = 0x5be0cd19;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t t1 = h + sigma1(e) + ch_lop3(e,f,g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj_lop3(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t w_new = gamma1(W[(i-2) & 15]) + W[(i-7) & 15] + gamma0(W[(i-15) & 15]) + W[i & 15];
        W[i & 15] = w_new;
        uint32_t t1 = h + sigma1(e) + ch_lop3(e,f,g) + K[i] + w_new;
        uint32_t t2 = sigma0(a) + maj_lop3(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    uint32_t hash[8];
    hash[0] = swap32(0x6a09e667 + a); hash[1] = swap32(0xbb67ae85 + b);
    hash[2] = swap32(0x3c6ef372 + c); hash[3] = swap32(0xa54ff53a + d);
    hash[4] = swap32(0x510e527f + e); hash[5] = swap32(0x9b05688c + f);
    hash[6] = swap32(0x1f83d9ab + g); hash[7] = swap32(0x5be0cd19 + h);

    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (hash[i] > s_target[i]) return;
        if (hash[i] < s_target[i]) break;
    }
    if (atomicCAS(result_found, 0, 1) == 0) *result_nonce = nonce;
}

//==============================================================================
// VARIANT 9: All Optimizations Combined
//==============================================================================

__global__ void mine_all_optimizations(
    const uint32_t* __restrict__ midstate,
    const uint32_t* __restrict__ tail,
    const uint32_t* __restrict__ target,
    uint32_t start_nonce,
    uint32_t* __restrict__ result_nonce,
    uint32_t* __restrict__ result_found
) {
    __shared__ uint32_t s_midstate[8];
    __shared__ uint32_t s_tail[4];
    __shared__ uint32_t s_target[8];

    if (threadIdx.x < 8) {
        s_midstate[threadIdx.x] = midstate[threadIdx.x];
        s_target[threadIdx.x] = target[threadIdx.x];
    }
    if (threadIdx.x < 4) s_tail[threadIdx.x] = tail[threadIdx.x];
    __syncthreads();

    if (*result_found) return;

    uint32_t nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t W[16];
    uint32_t a, b, c, d, e, f, g, h;

    uint32_t s0 = s_midstate[0], s1 = s_midstate[1], s2 = s_midstate[2], s3 = s_midstate[3];
    uint32_t s4 = s_midstate[4], s5 = s_midstate[5], s6 = s_midstate[6], s7 = s_midstate[7];

    // First SHA256 second block with sliding window + LOP3
    W[0] = swap32(s_tail[0]); W[1] = swap32(s_tail[1]); W[2] = swap32(s_tail[2]); W[3] = swap32(nonce);
    W[4] = 0x80000000; W[5] = 0; W[6] = 0; W[7] = 0;
    W[8] = 0; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 640;

    a = s0; b = s1; c = s2; d = s3; e = s4; f = s5; g = s6; h = s7;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t t1 = h + sigma1(e) + ch_lop3(e,f,g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj_lop3(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t w_new = gamma1(W[(i-2) & 15]) + W[(i-7) & 15] + gamma0(W[(i-15) & 15]) + W[i & 15];
        W[i & 15] = w_new;
        uint32_t t1 = h + sigma1(e) + ch_lop3(e,f,g) + K[i] + w_new;
        uint32_t t2 = sigma0(a) + maj_lop3(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    s0 += a; s1 += b; s2 += c; s3 += d; s4 += e; s5 += f; s6 += g; s7 += h;

    // Second SHA256 with precomputation + sliding + LOP3
    W[0] = s0; W[1] = s1; W[2] = s2; W[3] = s3;
    W[4] = s4; W[5] = s5; W[6] = s6; W[7] = s7;
    W[8] = 0x80000000; W[9] = 0; W[10] = 0; W[11] = 0;
    W[12] = 0; W[13] = 0; W[14] = 0; W[15] = 256;

    a = 0x6a09e667; b = 0xbb67ae85; c = 0x3c6ef372; d = 0xa54ff53a;
    e = 0x510e527f; f = 0x9b05688c; g = 0x1f83d9ab; h = 0x5be0cd19;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t t1 = h + sigma1(e) + ch_lop3(e,f,g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj_lop3(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t w_new = gamma1(W[(i-2) & 15]) + W[(i-7) & 15] + gamma0(W[(i-15) & 15]) + W[i & 15];
        W[i & 15] = w_new;
        uint32_t t1 = h + sigma1(e) + ch_lop3(e,f,g) + K[i] + w_new;
        uint32_t t2 = sigma0(a) + maj_lop3(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    // Early exit on MSW
    uint32_t h7 = swap32(0x5be0cd19 + h);
    if (h7 > s_target[7]) return;

    uint32_t hash[8];
    hash[7] = h7;
    hash[6] = swap32(0x1f83d9ab + g);
    hash[5] = swap32(0x9b05688c + f);
    hash[4] = swap32(0x510e527f + e);
    hash[3] = swap32(0xa54ff53a + d);
    hash[2] = swap32(0x3c6ef372 + c);
    hash[1] = swap32(0xbb67ae85 + b);
    hash[0] = swap32(0x6a09e667 + a);

    // Full target check for remaining words
    for (int i = 6; i >= 0; i--) {
        if (hash[i] > s_target[i]) return;
        if (hash[i] < s_target[i]) break;
    }
    if (atomicCAS(result_found, 0, 1) == 0) *result_nonce = nonce;
}

}  // namespace variants
}  // namespace tetsuo

#endif // TETSUO_SHA256_VARIANTS_CUH
