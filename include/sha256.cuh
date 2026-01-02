// SPDX-License-Identifier: MIT
// TETSUO GPU Miner - CUDA SHA256 Implementation
// Copyright (c) 2024-2026 TETSUO Contributors
//
// This implementation follows FIPS 180-4 (Secure Hash Standard)
// https://csrc.nist.gov/publications/detail/fips/180/4/final
//
// Optimized for CUDA with support for:
//   - Ampere (SM 80, 86)
//   - Ada/Lovelace (SM 89)
//   - Hopper (SM 90)
//   - Blackwell (SM 100, 120)

#ifndef TETSUO_SHA256_CUH
#define TETSUO_SHA256_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace tetsuo {

//==============================================================================
// SHA256 Constants (FIPS 180-4 Section 4.2.2)
//==============================================================================

// K[0..63] - First 32 bits of the fractional parts of the cube roots
// of the first 64 prime numbers (2..311)
__constant__ uint32_t K[64] = {
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

// H[0..7] - Initial hash values (FIPS 180-4 Section 5.3.3)
// First 32 bits of the fractional parts of the square roots
// of the first 8 prime numbers (2, 3, 5, 7, 11, 13, 17, 19)
__constant__ uint32_t H_INIT[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

//==============================================================================
// SHA256 Helper Functions (FIPS 180-4 Section 4.1.2)
//==============================================================================

/// Rotate right (circular right shift)
/// ROTR^n(x) = (x >> n) | (x << (32-n))
__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

/// Ch(x,y,z) = (x AND y) XOR (NOT x AND z)
/// "Choose" - for each bit, if x=1 choose y, else choose z
__device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

/// Maj(x,y,z) = (x AND y) XOR (x AND z) XOR (y AND z)
/// "Majority" - for each bit, return the majority value of x,y,z
__device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

/// Σ0(x) = ROTR^2(x) XOR ROTR^13(x) XOR ROTR^22(x)
/// Upper-case sigma 0, used in compression function
__device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

/// Σ1(x) = ROTR^6(x) XOR ROTR^11(x) XOR ROTR^25(x)
/// Upper-case sigma 1, used in compression function
__device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

/// σ0(x) = ROTR^7(x) XOR ROTR^18(x) XOR SHR^3(x)
/// Lower-case sigma 0, used in message schedule
__device__ __forceinline__ uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

/// σ1(x) = ROTR^17(x) XOR ROTR^19(x) XOR SHR^10(x)
/// Lower-case sigma 1, used in message schedule
__device__ __forceinline__ uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

//==============================================================================
// Endianness Conversion
//==============================================================================

/// Swap byte order (big-endian <-> little-endian)
/// Uses PTX byte permute instruction for efficiency
__device__ __forceinline__ uint32_t swap32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

//==============================================================================
// SHA256 Block Header Mining
//==============================================================================

/// Compute double SHA256 hash of 80-byte Bitcoin block header
///
/// Bitcoin uses SHA256d (double SHA256) for proof-of-work:
///   hash = SHA256(SHA256(header))
///
/// Block header format (80 bytes):
///   - Version (4 bytes)
///   - Previous block hash (32 bytes)
///   - Merkle root (32 bytes)
///   - Timestamp (4 bytes)
///   - Difficulty bits (4 bytes)
///   - Nonce (4 bytes) <- varied by miner
///
/// Endianness:
///   - SHA256 operates on big-endian 32-bit words
///   - Bitcoin interprets the 256-bit hash as little-endian integer
///   - Output is byte-swapped for correct numeric comparison with target
///
/// @param header Block header as 20 uint32_t words (little-endian)
/// @param nonce  Nonce value to try (replaces bytes 76-79)
/// @param hash   Output 256-bit hash as 8 uint32_t words (little-endian integer)
__device__ void sha256_block_header(const uint32_t* header, uint32_t nonce, uint32_t* hash) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t t1, t2;

    //--------------------------------------------------------------------------
    // First SHA256: Process 80-byte header
    //--------------------------------------------------------------------------

    // Initialize hash state
    uint32_t state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[i] = H_INIT[i];
    }

    //--- Block 1: bytes 0-63 (16 words) ---

    // Load first 64 bytes, convert to big-endian
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = swap32(header[i]);
    }

    // Extend message schedule (FIPS 180-4 Section 6.2.2)
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    }

    // Initialize working variables
    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    // 64 rounds of compression
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        t1 = h + sigma1(e) + ch(e, f, g) + K[i] + W[i];
        t2 = sigma0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    // Add compressed chunk to hash state
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;

    //--- Block 2: bytes 64-79 + padding ---

    // Last 16 bytes of header (with nonce at position 76-79)
    W[0] = swap32(header[16]);
    W[1] = swap32(header[17]);
    W[2] = swap32(header[18]);
    W[3] = swap32(nonce);      // Nonce position
    W[4] = 0x80000000;         // Padding: 1 bit followed by zeros
    #pragma unroll
    for (int i = 5; i < 15; i++) {
        W[i] = 0;
    }
    W[15] = 640;               // Message length: 80 bytes = 640 bits

    // Extend and compress
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    }

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        t1 = h + sigma1(e) + ch(e, f, g) + K[i] + W[i];
        t2 = sigma0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;

    //--------------------------------------------------------------------------
    // Second SHA256: Hash the 32-byte result
    //--------------------------------------------------------------------------

    uint32_t state2[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state2[i] = H_INIT[i];
    }

    // Input is the 32-byte hash from first SHA256
    // Note: No endian swap - already in big-endian from first hash
    W[0] = state[0]; W[1] = state[1]; W[2] = state[2]; W[3] = state[3];
    W[4] = state[4]; W[5] = state[5]; W[6] = state[6]; W[7] = state[7];
    W[8] = 0x80000000;         // Padding
    #pragma unroll
    for (int i = 9; i < 15; i++) {
        W[i] = 0;
    }
    W[15] = 256;               // Message length: 32 bytes = 256 bits

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    }

    a = state2[0]; b = state2[1]; c = state2[2]; d = state2[3];
    e = state2[4]; f = state2[5]; g = state2[6]; h = state2[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        t1 = h + sigma1(e) + ch(e, f, g) + K[i] + W[i];
        t2 = sigma0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    //--------------------------------------------------------------------------
    // Output: Convert to little-endian 256-bit integer
    //--------------------------------------------------------------------------
    // SHA256 produces output in big-endian word format.
    // Bitcoin interprets the 256-bit hash as a little-endian integer where:
    //   - bytes 0-3 (from state[0]) are the LEAST significant
    //   - bytes 28-31 (from state[7]) are the MOST significant
    //
    // The byte-swap converts from SHA256's big-endian to the target's
    // native byte order. After this, hash[0]=LSW and hash[7]=MSW for
    // correct numeric comparison with the difficulty target.
    hash[0] = swap32(state2[0] + a);
    hash[1] = swap32(state2[1] + b);
    hash[2] = swap32(state2[2] + c);
    hash[3] = swap32(state2[3] + d);
    hash[4] = swap32(state2[4] + e);
    hash[5] = swap32(state2[5] + f);
    hash[6] = swap32(state2[6] + g);
    hash[7] = swap32(state2[7] + h);
}

//==============================================================================
// Target Comparison
//==============================================================================

/// Check if hash meets difficulty target (hash <= target)
///
/// Both hash and target use Bitcoin's little-endian 256-bit integer convention:
///   - index 0 = LSW (least significant word, bits 0-31)
///   - index 7 = MSW (most significant word, bits 224-255)
///
/// Compare from MSW (index 7) to LSW (index 0) for correct numeric comparison.
/// This matches Bitcoin Core's arith_uint256::CompareTo() implementation.
///
/// @param hash   256-bit hash as 8 uint32_t words (little-endian integer)
/// @param target 256-bit target as 8 uint32_t words (little-endian integer)
/// @return true if hash <= target (valid proof-of-work)
__device__ bool check_target(const uint32_t* hash, const uint32_t* target) {
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (hash[i] < target[i]) return true;
        if (hash[i] > target[i]) return false;
    }
    return true;  // Equal means valid
}

//==============================================================================
// Mining Kernel
//==============================================================================

/// CUDA kernel to scan nonce range for valid blocks
///
/// Each thread tests one nonce value. The kernel uses an early-exit pattern
/// with atomic operations to handle the race condition when multiple threads
/// find valid blocks simultaneously.
///
/// @param header       80-byte block header (20 uint32_t, little-endian)
/// @param target       256-bit target (8 uint32_t, little-endian)
/// @param start_nonce  First nonce to test
/// @param result_nonce Output: winning nonce value
/// @param result_found Output: 1 if found, 0 otherwise (atomic)
__global__ void mine_kernel(
    const uint32_t* __restrict__ header,
    const uint32_t* __restrict__ target,
    uint32_t start_nonce,
    uint32_t* __restrict__ result_nonce,
    uint32_t* __restrict__ result_found
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = start_nonce + idx;

    // Early exit if solution already found
    if (*result_found) return;

    uint32_t hash[8];
    sha256_block_header(header, nonce, hash);

    if (check_target(hash, target)) {
        // Atomic CAS to handle race condition
        if (atomicCAS(result_found, 0, 1) == 0) {
            *result_nonce = nonce;
        }
    }
}

}  // namespace tetsuo

#endif // TETSUO_SHA256_CUH
