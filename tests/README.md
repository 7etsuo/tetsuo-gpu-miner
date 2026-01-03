# TETSUO GPU Miner - Test Suite

Test and benchmark suite for SHA256 mining kernel optimizations.

## Hardware Note

These benchmarks were developed and tested on an **RTX 3090 (Ampere, SM 8.6)**.
Performance characteristics will differ on other architectures. If you are using
RTX Pro 6000 Blackwell or other GPUs, re-run the benchmarks on your target hardware
to determine optimal configurations.

## Building

From the `build/` directory:

```bash
cmake ..
make test-sha256      # Correctness tests
make benchmark        # Basic benchmark (baseline vs optimized)
make benchmark-all    # Comprehensive benchmark of all variants
```

## Running Tests

### Correctness Tests

Verifies that all SHA256 implementations produce correct results:

```bash
./test-sha256
```

Tests include:
- Known SHA256 test vectors
- Bitcoin block header verification (block #125552)
- Random nonce mining validation
- Target comparison logic

### Basic Benchmark

Compares baseline kernel against midstate-optimized kernel:

```bash
./benchmark
```

### Comprehensive Benchmark

Tests all optimization variants and combinations:

```bash
./benchmark-all
```

## Benchmark Results (RTX 3090)

```
+----------------------------------+------------+----------+
| Optimization                     | Hash Rate  | Speedup  |
+----------------------------------+------------+----------+
| Baseline (no midstate)           | 2605.84 MH |   1.00x  |
| Midstate only                    | 4139.73 MH |   1.59x  |
| LOP3 instructions                | 4138.23 MH |   1.59x  |
| Early MSW exit                   | 4147.22 MH |   1.59x  |
| Sliding window W[16]             | 4129.58 MH |   1.58x  |
| Shared memory                    | 4080.05 MH |   1.57x  |
| SHA256-2 precompute              | 4110.89 MH |   1.58x  |
| LOP3 + Sliding                   | 4124.12 MH |   1.58x  |
| LOP3 + Shared                    | 4090.97 MH |   1.57x  |
| LOP3 + Sliding + Shared          | 4103.11 MH |   1.57x  |
| ALL OPTIMIZATIONS                | 4129.04 MH |   1.58x  |
+----------------------------------+------------+----------+
```

### Key Findings

1. **Midstate precomputation is the dominant optimization** - provides ~59% speedup alone
2. **Micro-optimizations don't stack** - LOP3, sliding window, shared memory achieve similar results
3. **Combinations show no synergy** - combined optimizations don't exceed individual ones
4. **Early MSW exit wins marginally** - 4147 MH/s, essentially within measurement noise

## Using the Optimized Kernel

To use the optimized kernel in your code:

```cpp
#include "sha256_optimized.cuh"

// Compute midstate on CPU (once per work unit)
uint32_t header[20];  // 80-byte block header as uint32_t array
uint32_t midstate[8];
tetsuo::optimized::compute_midstate_cpu(header, midstate);

// Prepare tail (last 16 bytes: merkle_root[28:32], time, bits, nonce_placeholder)
uint32_t tail[4] = {header[16], header[17], header[18], 0};

// Copy to device and launch
cudaMemcpy(d_midstate, midstate, 32, cudaMemcpyHostToDevice);
cudaMemcpy(d_tail, tail, 16, cudaMemcpyHostToDevice);
cudaMemcpy(d_target, target, 32, cudaMemcpyHostToDevice);

tetsuo::optimized::mine_kernel_optimized<<<grid, block>>>(
    d_midstate, d_tail, d_target, start_nonce, d_result_nonce, d_result_found
);
```

## Files

- `test_sha256.cu` - Correctness test suite
- `benchmark.cu` - Basic performance benchmark
- `benchmark_all.cu` - Comprehensive optimization benchmark
