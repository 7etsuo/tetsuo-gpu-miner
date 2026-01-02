// SPDX-License-Identifier: MIT
// TETSUO GPU Miner - Mining Structures and Functions
// Copyright (c) 2024-2026 TETSUO Contributors

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tetsuo
{

/// Work unit for mining
struct MiningWork
{
  std::vector<uint8_t> header; // 80-byte block header
  uint32_t target[8];          // 256-bit target (little-endian)
  std::string block_hex;       // Full block for submission
  int height = 0;              // Block height
  bool valid = false;          // Work validity flag
};

/// Convert compact difficulty bits to 256-bit target
///
/// Bitcoin's compact "bits" format encodes a 256-bit target as:
///   bits = 0xEEMMMMM (E=exponent, M=mantissa)
///   target = mantissa * 256^(exponent-3)
///
/// The target array uses little-endian 256-bit integer convention:
///   - target[0] = LSW (least significant word, bits 0-31)
///   - target[7] = MSW (most significant word, bits 224-255)
///
/// @param bits Compact difficulty bits from block header
/// @param target Output 8-word (256-bit) target array
void bits_to_target (uint32_t bits, uint32_t *target);

/// Build coinbase transaction for mining
///
/// Creates a valid coinbase transaction with:
///   - BIP34 height in scriptSig
///   - 8-byte extra nonce for uniqueness
///   - P2PKH output to specified address
///
/// @param height Block height (for BIP34 compliance)
/// @param value Block reward in satoshis
/// @param address Mining payout address (P2PKH)
/// @return Serialized transaction as hex string
std::string build_coinbase (int height, int64_t value,
                            const std::string &address);

/// Compute merkle root from transactions
///
/// Implements Bitcoin's merkle tree:
///   1. Hash all transactions with SHA256d
///   2. Pair and hash until single root remains
///   3. Odd-length levels duplicate last element
///
/// @param coinbase_hex Coinbase transaction (hex)
/// @param tx_hashes Transaction IDs (display order, will be reversed
/// internally)
/// @return Merkle root in internal byte order (NOT display order)
std::string compute_merkle_root (const std::string &coinbase_hex,
                                 const std::vector<std::string> &tx_hashes);

} // namespace tetsuo
