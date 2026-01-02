// SPDX-License-Identifier: MIT
// TETSUO GPU Miner - Configuration
// Copyright (c) 2024-2026 TETSUO Contributors

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tetsuo
{

/// Miner configuration
struct Config
{
  std::string rpc_url = "http://127.0.0.1:8337";
  std::string rpc_user;
  std::string rpc_pass;
  std::string mining_address;
  std::vector<int> devices; // Empty = use all available GPUs
  int block_size = 256;
  bool verbose = false;
};

// Named constants
namespace constants
{
constexpr int RPC_TIMEOUT_SECONDS = 30;
constexpr int WORK_RETRY_SECONDS = 5;
constexpr int STATS_INTERVAL_SECONDS = 1;
constexpr uint32_t NONCE_SPACE_PER_GPU = 0x40000000; // 2^30 nonces per GPU
constexpr size_t BLOCK_HEADER_SIZE = 80;
constexpr int DEFAULT_RPC_PORT = 8337;
constexpr int MAX_GPUS = 16;
constexpr const char *VERSION = "2.0.0";
}

} // namespace tetsuo
